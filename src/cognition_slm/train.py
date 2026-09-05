"""Train the tiny coding/cognition language model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path

from .checkpoint import load_checkpoint_payload
from .config import MODEL_PRESETS, ModelConfig
from .data import encode_examples, load_jsonl
from .tokenizer import ByteTokenizer


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install project dependencies with: "
            "python -m pip install \".[dev]\""
        ) from exc
    return torch


def _device(torch, name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _move_optimizer_state(torch, optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _batch(torch, encoded, indices, device):
    max_length = max(len(encoded[index]["input_ids"]) for index in indices)
    input_ids = []
    attention_mask = []
    task_labels = []
    error_labels = []
    confidence_labels = []
    pool_positions = []
    lm_loss_masks = []
    for index in indices:
        item = encoded[index]
        padding = [0] * (max_length - len(item["input_ids"]))
        input_ids.append(item["input_ids"] + padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * len(padding))
        task_labels.append(item["task_label"])
        error_labels.append(item["error_label"])
        confidence_labels.append(item["confidence_label"])
        pool_positions.append(item["pool_position"])
        lm_loss_masks.append(
            [
                int(target_position >= item["answer_start"])
                for target_position in range(1, len(item["input_ids"]))
            ]
            + [0] * max(0, max_length - len(item["input_ids"])),
        )
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(task_labels, dtype=torch.long, device=device),
        torch.tensor(error_labels, dtype=torch.long, device=device),
        torch.tensor(confidence_labels, dtype=torch.long, device=device),
        torch.tensor(pool_positions, dtype=torch.long, device=device),
        torch.tensor(lm_loss_masks, dtype=torch.bool, device=device),
    )


def _validation_summary(torch, model, encoded, batch_size: int, device) -> dict[str, float | int]:
    model.eval()
    auxiliary_loss_total = 0.0
    lm_loss_total = 0.0
    lm_token_count = 0
    task_correct = 0
    error_correct = 0
    confidence_correct = 0
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            indices = list(range(start, min(start + batch_size, len(encoded))))
            batch = _batch(torch, encoded, indices, device)
            output = model(
                batch[0],
                attention_mask=batch[1],
                task_labels=batch[2],
                error_labels=batch[3],
                confidence_labels=batch[4],
                pool_positions=batch[5],
                lm_loss_mask=batch[6],
            )
            if output.loss is None:
                raise RuntimeError("model returned no validation loss")
            count = len(indices)
            targets = int(batch[6].sum().item())
            if output.lm_loss is not None:
                lm_loss_total += float(output.lm_loss.detach().cpu()) * targets
                lm_token_count += targets
            for auxiliary in (output.task_loss, output.error_loss, output.confidence_loss):
                if auxiliary is not None:
                    auxiliary_loss_total += 0.25 * float(auxiliary.detach().cpu()) * count
            task_correct += int((output.task_logits.argmax(dim=-1) == batch[2]).sum().item())
            error_correct += int((output.error_logits.argmax(dim=-1) == batch[3]).sum().item())
            confidence_correct += int(
                (output.confidence_logits.argmax(dim=-1) == batch[4]).sum().item()
            )
    total = len(encoded)
    return {
        "records": total,
        "loss": lm_loss_total / max(1, lm_token_count) + auxiliary_loss_total / total,
        "lm_loss": lm_loss_total / max(1, lm_token_count),
        "supervised_tokens": lm_token_count,
        "task_accuracy": task_correct / total,
        "error_accuracy": error_correct / total,
        "confidence_bucket_accuracy": confidence_correct / total,
    }


def _atomic_save(torch, payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _optimizer_parameters(model, weight_decay: float, legacy: bool = False):
    if legacy:
        return model.parameters()
    return [
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
         "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
    ]


def _scaled_optimizer_step(scaler, optimizer, scheduler) -> bool:
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    succeeded = scaler.get_scale() >= previous_scale
    if succeeded:
        scheduler.step()
    return succeeded


def _runtime_options(args):
    precision = getattr(args, "precision", "fp32")
    accumulation = getattr(args, "gradient_accumulation_steps", 1)
    save_every = getattr(args, "save_every", 100)
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    if accumulation < 1 or save_every < 1:
        raise ValueError("gradient_accumulation_steps and save_every must be positive")
    return precision, accumulation, save_every


def train(args: argparse.Namespace) -> dict:
    precision, accumulation, save_every = _runtime_options(args)
    examples = load_jsonl(args.data)
    torch = None
    checkpoint = None
    if args.dry_run and args.resume:
        raise ValueError("--dry-run cannot be combined with --resume")
    if args.resume:
        torch = _import_torch()
        checkpoint, config = load_checkpoint_payload(torch, args.resume)
    else:
        preset = MODEL_PRESETS[getattr(args, "preset", "demo")]
        for name, value in preset.items():
            if getattr(args, name, None) is None:
                setattr(args, name, value)
        config = ModelConfig(
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            architecture=args.architecture,
        )
    config.validate()
    tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
    encoded = encode_examples(examples, tokenizer, config.block_size)
    validation_encoded = None
    if args.eval_data:
        validation_examples = load_jsonl(args.eval_data)
        validation_encoded = encode_examples(validation_examples, tokenizer, config.block_size)
    if args.dry_run:
        return {
            "records": len(examples),
            "max_tokens": max(len(item["input_ids"]) for item in encoded),
            "block_size": config.block_size,
            "architecture": config.architecture,
            "truncated_records": sum(bool(item.get("truncated", False)) for item in encoded),
            "status": "dry-run",
        }

    if torch is None:
        torch = _import_torch()
    from .model import CognitionSLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(torch, args.device)
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16 and bf16 training require a CUDA device; use --precision fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16; use --precision fp16 or fp32")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model = CognitionSLM(config).to(device)
    model.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)
    previous_optimizer = checkpoint.get("optimizer_state_dict") if checkpoint else None
    legacy_groups = isinstance(previous_optimizer, dict) and len(previous_optimizer.get("param_groups", [])) == 1
    optimizer = torch.optim.AdamW(
        _optimizer_parameters(model, args.weight_decay, legacy_groups),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    start_step = 0
    base_learning_rate = args.learning_rate
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if isinstance(metadata.get("learning_rate"), (int, float)):
            base_learning_rate = float(metadata["learning_rate"])
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state(torch, optimizer, device)
            start_step = int(metadata.get("step", 0))
    if start_step >= args.steps:
        raise ValueError(f"--steps must exceed checkpoint step {start_step}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_scale(step, args.steps, args.warmup_steps),
    )
    if checkpoint is not None and isinstance(checkpoint.get("scheduler_state_dict"), dict):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if start_step:
        # AMP may skip optimizer updates, so scheduler progress can lag batch steps.
        schedule_step = scheduler.last_epoch if isinstance(
            checkpoint.get("scheduler_state_dict"), dict
        ) else start_step
        next_learning_rate = base_learning_rate * _lr_scale(
            schedule_step, args.steps, args.warmup_steps
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = next_learning_rate
    if checkpoint is not None:
        if checkpoint.get("scaler_state_dict") and precision == "fp16":
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if isinstance(checkpoint.get("torch_rng_state"), torch.Tensor):
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng = checkpoint.get("cuda_rng_state_all")
        if device.type == "cuda" and isinstance(cuda_rng, list) and cuda_rng:
            if len(cuda_rng) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA device count differs; exact RNG resume is unavailable")
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    model.train()
    output_path = Path(args.out)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    history: list[float] = []
    validation_history: list[dict[str, float | int]] = []

    def save(step: int, final_loss: float) -> None:
        _atomic_save(torch, {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "metadata": {
                "records": len(examples), "steps": args.steps, "step": step,
                "seed": args.seed, "device": str(device),
                "learning_rate": base_learning_rate, "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps, "resumed_from_step": start_step,
                "final_loss": final_loss, "validation_history": validation_history,
                "precision": precision, "gradient_accumulation_steps": accumulation,
                "parameter_count": parameter_count,
                "effective_batch_size": args.batch_size * accumulation,
                "optimizer_steps": scheduler.last_epoch,
                "skipped_optimizer_steps": skipped_optimizer_steps,
            },
        }, output_path)

    for step in range(start_step + 1, args.steps + 1):
        step_rng = random.Random(args.seed + step)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        microbatches = [
            [step_rng.randrange(len(encoded)) for _ in range(args.batch_size)]
            for _ in range(accumulation)
        ]
        total_targets = sum(
            len(encoded[index]["input_ids"]) - encoded[index]["answer_start"]
            for indices in microbatches for index in indices
        )
        for indices in microbatches:
            batch = _batch(torch, encoded, indices, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
                output = model(
                    batch[0], attention_mask=batch[1], task_labels=batch[2],
                    error_labels=batch[3], confidence_labels=batch[4],
                    pool_positions=batch[5], lm_loss_mask=batch[6],
                )
                if output.loss is None:
                    raise RuntimeError("model returned no loss")
                if output.lm_loss is None:
                    raise RuntimeError("model returned no language-model loss")
                targets = sum(
                    len(encoded[index]["input_ids"]) - encoded[index]["answer_start"]
                    for index in indices
                )
                # Match one combined batch despite unequal answer lengths.
                loss = output.lm_loss * (targets / total_targets)
                for auxiliary in (output.task_loss, output.error_loss, output.confidence_loss):
                    if auxiliary is not None:
                        loss = loss + 0.25 * auxiliary / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {step}")
            scaler.scale(loss).backward()
            loss_value += float(loss.detach().cpu())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=precision != "fp16"
        )
        if _scaled_optimizer_step(scaler, optimizer, scheduler):
            successful_optimizer_steps += 1
        else:
            skipped_optimizer_steps += 1
        history.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step={step} loss={loss_value:.4f} device={device}")
        if validation_encoded is not None and (
            step % args.eval_every == 0 or step == args.steps
        ):
            validation = _validation_summary(
                torch, model, validation_encoded, args.batch_size, device
            )
            validation["step"] = step
            validation_history.append(validation)
            print(
                f"eval_step={step} val_loss={validation['loss']:.4f} "
                f"task_accuracy={validation['task_accuracy']:.3f}"
            )
            model.train()

        if step % save_every == 0 or step == args.steps:
            save(step, loss_value)

    if successful_optimizer_steps == 0:
        raise RuntimeError(
            "training completed no optimizer updates; checkpoint saved for diagnosis. "
            "Reduce the learning rate or use --precision fp32"
        )
    final_loss = history[-1]
    return {
        "records": len(examples),
        "steps": args.steps,
        "final_loss": final_loss,
        "checkpoint": str(output_path),
        "device": str(device),
        "resumed_from_step": start_step,
        "validation": validation_history[-1] if validation_history else None,
        "parameter_count": parameter_count,
        "effective_batch_size": args.batch_size * accumulation,
        "optimizer_steps": scheduler.last_epoch,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "max_tokens": max(len(item["input_ids"]) for item in encoded),
        "block_size": config.block_size,
        "precision": precision,
        "truncated_records": sum(bool(item.get("truncated", False)) for item in encoded),
        "status": "trained",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="artifacts/demo.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--architecture", choices=("modern", "legacy"))
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-data")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preset", choices=tuple(MODEL_PRESETS), default="demo")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name, value in MODEL_PRESETS[args.preset].items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    try:
        _runtime_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.n_layer < 1 or args.n_head < 1 or args.n_embd < 1:
        parser.error("--n-layer, --n-head, and --n-embd must be positive")
    if args.weight_decay < 0 or args.warmup_steps < 0 or args.eval_every < 1:
        parser.error("--weight-decay must be non-negative; --warmup-steps and --eval-every must be positive")
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.log_every < 1:
        parser.error("--learning-rate and --grad-clip must be positive; --log-every must be positive")
    if args.warmup_steps > args.steps:
        parser.error("--warmup-steps cannot exceed --steps")
    if args.dry_run and args.resume:
        parser.error("--dry-run cannot be combined with --resume")
    if args.eval_data and Path(args.eval_data).resolve() == Path(args.data).resolve():
        parser.error("--eval-data must be different from --data")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
