"""Train the tiny coding/cognition language model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .checkpoint import load_checkpoint_payload
from .config import ModelConfig
from .data import encode_examples, load_jsonl
from .tokenizer import ByteTokenizer


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install project dependencies with: "
            "python -m pip install -e ."
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
    for index in indices:
        item = encoded[index]
        padding = [0] * (max_length - len(item["input_ids"]))
        input_ids.append(item["input_ids"] + padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * len(padding))
        task_labels.append(item["task_label"])
        error_labels.append(item["error_label"])
        confidence_labels.append(item["confidence_label"])
        pool_positions.append(item["pool_position"])
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(task_labels, dtype=torch.long, device=device),
        torch.tensor(error_labels, dtype=torch.long, device=device),
        torch.tensor(confidence_labels, dtype=torch.long, device=device),
        torch.tensor(pool_positions, dtype=torch.long, device=device),
    )


def _validation_summary(torch, model, encoded, batch_size: int, device) -> dict[str, float | int]:
    model.eval()
    loss_total = 0.0
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
            )
            if output.loss is None:
                raise RuntimeError("model returned no validation loss")
            count = len(indices)
            loss_total += float(output.loss.detach().cpu()) * count
            task_correct += int((output.task_logits.argmax(dim=-1) == batch[2]).sum().item())
            error_correct += int((output.error_logits.argmax(dim=-1) == batch[3]).sum().item())
            confidence_correct += int(
                (output.confidence_logits.argmax(dim=-1) == batch[4]).sum().item()
            )
    total = len(encoded)
    return {
        "records": total,
        "loss": loss_total / total,
        "task_accuracy": task_correct / total,
        "error_accuracy": error_correct / total,
        "confidence_bucket_accuracy": confidence_correct / total,
    }


def train(args: argparse.Namespace) -> dict:
    examples = load_jsonl(args.data)
    torch = None
    checkpoint = None
    if args.resume:
        torch = _import_torch()
        checkpoint, config = load_checkpoint_payload(torch, args.resume)
    else:
        config = ModelConfig(block_size=args.block_size)
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
    model = CognitionSLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
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
    model.train()
    history: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    for step in range(start_step + 1, args.steps + 1):
        step_rng = random.Random(args.seed + step)
        indices = [step_rng.randrange(len(encoded)) for _ in range(args.batch_size)]
        batch = _batch(torch, encoded, indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch[0],
            attention_mask=batch[1],
            task_labels=batch[2],
            error_labels=batch[3],
            confidence_labels=batch[4],
            pool_positions=batch[5],
        )
        if output.loss is None:
            raise RuntimeError("model returned no loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        loss_value = float(output.loss.detach().cpu())
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

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_loss = history[-1] if history else float("nan")
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metadata": {
                "records": len(examples),
                "steps": args.steps,
                "step": args.steps,
                "seed": args.seed,
                "device": str(device),
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "resumed_from_step": start_step,
                "final_loss": final_loss,
                "validation_history": validation_history,
            },
        },
        output_path,
    )
    return {
        "records": len(examples),
        "steps": args.steps,
        "final_loss": final_loss,
        "checkpoint": str(output_path),
        "device": str(device),
        "resumed_from_step": start_step,
        "validation": validation_history[-1] if validation_history else None,
        "status": "trained",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="artifacts/demo.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
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
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
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
