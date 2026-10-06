"""Train the tiny coding/cognition language model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path
from time import monotonic

from .checkpoint import load_checkpoint_payload
from .config import MODEL_PRESETS, ModelConfig
from .data import encode_examples, file_sha256, load_jsonl, load_pretrain_text, pack_pretrain_text
from .tokenizer import ByteTokenizer

DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_AUX_LOSS_WEIGHT = 0.25
DEFAULT_WARMUP_STEPS = 5
DEFAULT_WEIGHT_DECAY = 0.01


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install project dependencies with: "
            "python -m pip install \".[dev]\""
        ) from exc
    return torch


def _lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    # Decay over the full post-warmup span so the final update still gets a nonzero rate.
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@lru_cache(maxsize=2)
def _epoch_permutation(seed: int, dataset_size: int, epoch: int) -> tuple[int, ...]:
    permutation = list(range(dataset_size))
    random.Random(f"{seed}:{epoch}").shuffle(permutation)
    return tuple(permutation)


def _sample_indices(seed: int, dataset_size: int, start: int, count: int) -> list[int]:
    """Read `count` indices from seeded per-epoch permutations at global sample `start`."""
    return [
        _epoch_permutation(seed, dataset_size, position // dataset_size)[position % dataset_size]
        for position in range(start, start + count)
    ]


def _restore_cuda_rng(torch, checkpoint: dict, device) -> None:
    states = checkpoint.get("cuda_rng_state_all")
    if device.type != "cuda" or not isinstance(states, list) or not states:
        return
    # Training uses one GPU. Older checkpoints stored every visible GPU's state, so pick
    # the training device's entry instead of requiring the same GPU count on resume.
    metadata = checkpoint.get("metadata")
    saved_device = str(metadata.get("device", "cuda")) if isinstance(metadata, dict) else "cuda"
    index = int(saved_device.split(":")[1]) if ":" in saved_device else 0
    torch.cuda.set_rng_state(states[min(index, len(states) - 1)].cpu(), device)


def _skip_nonfinite_update(precision: str, scaler, optimizer, step: int) -> None:
    if precision != "fp16":
        raise RuntimeError(f"non-finite training loss at step {step}")
    # fp16 overflow is expected occasionally; back off the scale as GradScaler would.
    optimizer.zero_grad(set_to_none=True)
    scaler.update(new_scale=scaler.get_scale() / 2)


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
    # Packed pretraining rows have no labels; None makes the model skip auxiliary losses.
    def labels(values):
        if any(value is None for value in values):
            return None
        return torch.tensor(values, dtype=torch.long, device=device)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        labels(task_labels),
        labels(error_labels),
        labels(confidence_labels),
        torch.tensor(pool_positions, dtype=torch.long, device=device),
        torch.tensor(lm_loss_masks, dtype=torch.bool, device=device),
    )


def _validation_summary(
    torch, model, encoded, batch_size: int, device, aux_loss_weight: float = DEFAULT_AUX_LOSS_WEIGHT,
) -> dict[str, float | int | None]:
    model.eval()
    auxiliary_loss_total = 0.0
    lm_loss_total = 0.0
    lm_token_count = 0
    task_correct = 0
    error_correct = 0
    confidence_correct = 0
    labeled = True
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
                    auxiliary_loss_total += aux_loss_weight * float(auxiliary.detach().cpu()) * count
            if batch[2] is None:
                labeled = False
                continue
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
        "task_accuracy": task_correct / total if labeled else None,
        "error_accuracy": error_correct / total if labeled else None,
        "confidence_bucket_accuracy": confidence_correct / total if labeled else None,
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


def _training_source_changes(metadata: dict, current: dict) -> list[str] | None:
    """Recorded fields that differ from this run, or None for checkpoints without a data fingerprint."""
    changes = [f"{key} {metadata[key]!r} -> {value!r}" for key, value in current.items()
               if key in metadata and metadata[key] != value]
    return changes if "training_source_sha256" in metadata else None


def _encode(path: str, examples, tokenizer: ByteTokenizer, config: ModelConfig) -> list[dict]:
    try:
        return encode_examples(examples, tokenizer, config.block_size, task_types=config.task_types,
                               error_categories=config.error_categories)
    except ValueError as exc:
        # encode_examples names only the record, so a bad --eval-data row would read as a --data error.
        raise type(exc)(f"{path}: {exc}") from None


def _runtime_options(args):
    precision = getattr(args, "precision", "fp32")
    accumulation = getattr(args, "gradient_accumulation_steps", 1)
    save_every = getattr(args, "save_every", 100)
    max_seconds = getattr(args, "max_seconds", None)
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    aux_loss_weight = getattr(args, "aux_loss_weight", None)
    if aux_loss_weight is not None and (not math.isfinite(aux_loss_weight) or aux_loss_weight < 0):
        raise ValueError("aux_loss_weight must be non-negative and finite")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    if accumulation < 1 or save_every < 1:
        raise ValueError("gradient_accumulation_steps and save_every must be positive")
    return precision, accumulation, save_every


def train(args: argparse.Namespace) -> dict:
    precision, accumulation, save_every = _runtime_options(args)
    aux_loss_weight = getattr(args, "aux_loss_weight", None)
    warmup_steps = getattr(args, "warmup_steps", None)
    weight_decay = getattr(args, "weight_decay", None)
    pretrain_text = getattr(args, "pretrain_text", None)
    pretrain_eval_text = getattr(args, "pretrain_eval_text", None)
    if (pretrain_text is None) == (getattr(args, "data", None) is None):
        raise ValueError("pass exactly one of --data or --pretrain-text")
    if pretrain_eval_text and args.eval_data:
        raise ValueError("--eval-data cannot be combined with --pretrain-eval-text")
    if args.dry_run and args.resume:
        raise ValueError("--dry-run cannot be combined with --resume")
    best_out = getattr(args, "best_out", None)
    if best_out and not (args.eval_data or pretrain_eval_text):
        raise ValueError("--best-out requires --eval-data or --pretrain-eval-text")
    torch = None
    fused_adamw = getattr(args, "fused_adamw", False)
    # Checked before the data and any resume checkpoint load, which can take a while on a full corpus.
    # A dry run never picks a device, so it still works without torch.
    if not args.dry_run:
        torch = _import_torch()
        from .generate import _device

        device = _device(args.device)
        if fused_adamw and device.type != "cuda":
            raise ValueError("--fused-adamw requires a CUDA device")
        if precision != "fp32" and device.type != "cuda":
            raise ValueError("fp16 and bf16 training require a CUDA device; use --precision fp32")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bf16; use --precision fp16 or fp32")
    # For raw text, records counts documents; packed rows are reported separately.
    examples = load_pretrain_text(pretrain_text) if pretrain_text else load_jsonl(args.data)
    checkpoint = None
    if args.resume:
        checkpoint, config = load_checkpoint_payload(torch, args.resume)
        saved_metadata = checkpoint.get("metadata")
        saved_weight = saved_metadata.get("aux_loss_weight") if isinstance(saved_metadata, dict) else None
        # An omitted flag keeps the resumed run's weight; older checkpoints predate it and used 0.25.
        if aux_loss_weight is None and isinstance(saved_weight, (int, float)):
            aux_loss_weight = float(saved_weight)
        # Likewise the warmup, or a resumed run would rejoin its schedule at the default warmup's rate.
        saved_warmup = saved_metadata.get("warmup_steps") if isinstance(saved_metadata, dict) else None
        if warmup_steps is None and isinstance(saved_warmup, int):
            warmup_steps = saved_warmup
            # main checks only an explicit flag, and a warmup past --steps would never reach the decay.
            if warmup_steps > args.steps:
                raise ValueError(f"the checkpoint's warmup of {warmup_steps} steps exceeds --steps {args.steps}; "
                                 "pass a shorter --warmup-steps")
        # And the weight decay, or the default would replace the decay the optimizer state was trained with.
        saved_decay = saved_metadata.get("weight_decay") if isinstance(saved_metadata, dict) else None
        if weight_decay is None and isinstance(saved_decay, (int, float)):
            weight_decay = float(saved_decay)
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
    if aux_loss_weight is None:
        aux_loss_weight = DEFAULT_AUX_LOSS_WEIGHT
    if warmup_steps is None:
        warmup_steps = DEFAULT_WARMUP_STEPS
    if weight_decay is None:
        weight_decay = DEFAULT_WEIGHT_DECAY
    tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
    if pretrain_text:
        encoded = pack_pretrain_text(examples, tokenizer, config.block_size)
    else:
        encoded = _encode(args.data, examples, tokenizer, config)
    objective = "pretrain" if pretrain_text else "sft"
    validation_encoded = None
    if args.eval_data:
        validation_examples = load_jsonl(args.eval_data)
        validation_encoded = _encode(args.eval_data, validation_examples, tokenizer, config)
    elif pretrain_eval_text:
        validation_encoded = pack_pretrain_text(
            load_pretrain_text(pretrain_eval_text), tokenizer, config.block_size
        )
    if args.dry_run:
        return {
            "records": len(examples),
            "objective": objective,
            "packed_rows": len(encoded),
            "max_tokens": max(len(item["input_ids"]) for item in encoded),
            "block_size": config.block_size,
            "architecture": config.architecture,
            "truncated_records": sum(bool(item.get("truncated", False)) for item in encoded),
            "status": "dry-run",
        }

    output_path = Path(args.out)
    best_path = Path(best_out) if best_out else None
    # A bad --out would otherwise fail at the first save, after the steps that save was meant to keep.
    for flag, path in (("--out", output_path), ("--best-out", best_path)):
        if path is None:
            continue
        if path.is_dir():
            raise ValueError(f"{flag} {path} is a directory; name a checkpoint file such as {path / 'model.pt'}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, probe = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            os.close(descriptor)
            Path(probe).unlink()
        except OSError as exc:
            raise ValueError(f"cannot write {flag} {path}: {exc}") from exc
    source_path = Path(pretrain_text or args.data).resolve()
    source_sha256 = file_sha256(source_path)
    data_changed = False
    if checkpoint is not None:
        recorded = checkpoint.get("metadata")
        recorded = recorded if isinstance(recorded, dict) else {}
        # The path is reported but not compared: the same file mounts at new paths across sessions.
        changes = _training_source_changes(recorded, {
            "objective": objective, "block_size": config.block_size,
            "records": len(examples), "training_source_sha256": source_sha256,
        })
        if changes is None:
            warnings.warn(
                f"{args.resume} predates training-source metadata, so its data cannot be compared "
                f"with {source_path}; resuming without the check", stacklevel=2,
            )
        elif changes:
            detail = (f"{'; '.join(changes)} (checkpoint trained on "
                      f"{recorded.get('training_source')}, this run reads {source_path})")
            if not getattr(args, "allow_data_change", False):
                raise ValueError(
                    f"--resume {args.resume} was trained on different data: {detail}. Pass "
                    "--allow-data-change to continue from its weights on this data with samples_seen reset to 0"
                )
            data_changed = True
            print(f"allow_data_change: {detail}; samples_seen reset to 0")

    from .model import CognitionSLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    model = CognitionSLM(config).to(device)
    model.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)
    previous_optimizer = checkpoint.get("optimizer_state_dict") if checkpoint else None
    legacy_groups = isinstance(previous_optimizer, dict) and len(previous_optimizer.get("param_groups", [])) == 1
    requested_learning_rate = getattr(args, "learning_rate", None)
    optimizer = torch.optim.AdamW(
        _optimizer_parameters(model, weight_decay, legacy_groups),
        lr=requested_learning_rate or DEFAULT_LEARNING_RATE, weight_decay=weight_decay,
        **({"fused": True} if fused_adamw else {}),
    )
    start_step = 0
    sample_offset = 0
    base_learning_rate = requested_learning_rate or DEFAULT_LEARNING_RATE
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if isinstance(metadata.get("learning_rate"), (int, float)):
            saved_learning_rate = float(metadata["learning_rate"])
            if getattr(args, "override_learning_rate", False):
                if requested_learning_rate is None:
                    raise ValueError("--override-learning-rate requires --learning-rate")
            elif requested_learning_rate is not None and not math.isclose(
                requested_learning_rate, saved_learning_rate, rel_tol=1e-9, abs_tol=0.0
            ):
                raise ValueError(
                    f"--learning-rate {requested_learning_rate:g} differs from checkpoint "
                    f"learning rate {saved_learning_rate:g}; omit it to keep the checkpoint "
                    "rate or pass --override-learning-rate to change the peak rate"
                )
            else:
                base_learning_rate = saved_learning_rate
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            # Loading restores the old run's decay; apply this run's decay so the update matches the metadata.
            # The first group is the decayed one in both the legacy and the split layout.
            optimizer.param_groups[0]["weight_decay"] = weight_decay
            # Resume restores old execution flags as well as Adam moments, so a fused run resumed without
            # --fused-adamw would keep stepping fused; this run's own choice replaces them either way.
            for group in optimizer.param_groups:
                group["fused"] = True if fused_adamw else None
                group["foreach"] = False if fused_adamw else None
            _move_optimizer_state(torch, optimizer, device)
            start_step = int(metadata.get("step", 0))
            samples_seen = metadata.get("samples_seen")
            sample_offset = int(samples_seen) if isinstance(samples_seen, int) else (
                start_step * args.batch_size * accumulation
            )
    if data_changed:
        # The old sample position indexes a permutation of different rows.
        sample_offset = 0
    best_step = None
    best_loss = None
    if checkpoint is not None and best_path is not None and not data_changed:
        # Carrying the resumed run's best keeps a worse later validation from replacing its better file,
        # but only for the same file that run wrote; any other path starts its own best.
        recorded_loss, recorded_out = metadata.get("best_eval_lm_loss"), metadata.get("best_out")
        if isinstance(recorded_loss, (int, float)) and isinstance(recorded_out, str) \
                and Path(recorded_out) == best_path.resolve() and best_path.exists():
            best_loss, best_step = float(recorded_loss), metadata.get("best_step")
    if checkpoint is not None and sample_offset:
        # samples_seen indexes the seed's epoch permutations; another seed would repeat and skip records.
        saved_seed = metadata.get("seed")
        if isinstance(saved_seed, int) and saved_seed != args.seed:
            raise ValueError(
                f"--seed {args.seed} differs from checkpoint seed {saved_seed}; pass --seed {saved_seed} "
                "to continue its sample stream without repeating or skipping records"
            )
    if start_step >= args.steps:
        raise ValueError(f"--steps must exceed checkpoint step {start_step}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_scale(step, args.steps, warmup_steps),
    )
    if checkpoint is not None and isinstance(checkpoint.get("scheduler_state_dict"), dict):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint is not None:
        # Restored optimizer and scheduler state carry the old peak rate; keep them in
        # line with the chosen one so later scheduler steps use it too.
        scheduler.base_lrs = [base_learning_rate] * len(optimizer.param_groups)
        for parameter_group in optimizer.param_groups:
            parameter_group["initial_lr"] = base_learning_rate
    if checkpoint is not None:
        # A step-0 checkpoint still carries the old lr from its optimizer state.
        # AMP may skip optimizer updates, so scheduler progress can lag batch steps.
        schedule_step = scheduler.last_epoch if isinstance(
            checkpoint.get("scheduler_state_dict"), dict
        ) else start_step
        next_learning_rate = base_learning_rate * _lr_scale(
            schedule_step, args.steps, warmup_steps
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = next_learning_rate
    if checkpoint is not None:
        if checkpoint.get("scaler_state_dict") and precision == "fp16":
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if isinstance(checkpoint.get("torch_rng_state"), torch.Tensor):
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        _restore_cuda_rng(torch, checkpoint, device)
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    history: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    max_seconds = getattr(args, "max_seconds", None)
    stopped_reason = "steps_completed"
    duration_seconds = 0.0
    processed_input_tokens = 0
    supervised_tokens = 0
    updated_supervised_tokens = 0

    def save(step: int, final_loss: float) -> None:
        _atomic_save(torch, {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            # Only the training device's state; the key name is kept for older readers.
            "cuda_rng_state_all": [torch.cuda.get_rng_state(device)] if device.type == "cuda" else [],
            "metadata": {
                "records": len(examples), "steps": step, "step": step,
                "objective": objective, "packed_rows": len(encoded),
                "training_source": str(source_path), "training_source_sha256": source_sha256,
                "block_size": config.block_size,
                "aux_loss_weight": aux_loss_weight,
                "requested_steps": args.steps, "completed_steps": step - start_step,
                "stopped_reason": stopped_reason, "duration_seconds": duration_seconds,
                "max_seconds": max_seconds,
                "fused_adamw": fused_adamw,
                "processed_input_tokens": processed_input_tokens,
                "supervised_tokens": supervised_tokens,
                "updated_supervised_tokens": updated_supervised_tokens,
                "token_counter_scope": "current invocation; non-padding UTF-8 byte tokens including special tokens",
                "seed": args.seed, "device": str(device),
                "learning_rate": base_learning_rate, "weight_decay": weight_decay,
                "warmup_steps": warmup_steps, "resumed_from_step": start_step,
                "final_loss": final_loss, "validation_history": validation_history,
                "precision": precision, "gradient_accumulation_steps": accumulation,
                "parameter_count": parameter_count,
                "effective_batch_size": args.batch_size * accumulation,
                "optimizer_steps": scheduler.last_epoch,
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "samples_seen": sample_offset,
                "best_step": best_step, "best_eval_lm_loss": best_loss,
                "best_out": str(best_path.resolve()) if best_path else None,
            },
        }, output_path)

    def save_best(step: int, validation: dict) -> None:
        # Weights only, the layout stage 4 ships, which every checkpoint loader reads.
        _atomic_save(torch, {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metadata": {
                "records": len(examples), "steps": step, "step": step, "objective": objective,
                "training_source": str(source_path), "training_source_sha256": source_sha256,
                "block_size": config.block_size, "seed": args.seed, "device": str(device),
                "learning_rate": base_learning_rate, "precision": precision,
                "parameter_count": parameter_count, "checkpoint_format": "model_only",
                "best_step": step, "best_eval_lm_loss": validation["lm_loss"], "validation": validation,
            },
        }, best_path)

    started_at = monotonic()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        # Epoch permutations cover every record once per pass and resume from samples_seen.
        drawn = _sample_indices(args.seed, len(encoded), sample_offset, args.batch_size * accumulation)
        sample_offset += len(drawn)
        microbatches = [
            drawn[start:start + args.batch_size] for start in range(0, len(drawn), args.batch_size)
        ]
        total_targets = sum(
            len(encoded[index]["input_ids"]) - encoded[index]["answer_start"]
            for indices in microbatches for index in indices
        )
        processed_input_tokens += sum(len(encoded[index]["input_ids"])
                                      for indices in microbatches for index in indices)
        supervised_tokens += total_targets
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
                        loss = loss + aux_loss_weight * auxiliary / accumulation
            scaler.scale(loss).backward()
            accumulated_loss += loss.detach()
        # Synchronize once per update instead of twice per microbatch.
        loss_value = float(accumulated_loss.cpu())
        if not math.isfinite(loss_value):
            _skip_nonfinite_update(precision, scaler, optimizer, step)
            skipped_optimizer_steps += 1
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=precision != "fp16"
            )
            if _scaled_optimizer_step(scaler, optimizer, scheduler):
                successful_optimizer_steps += 1
                updated_supervised_tokens += total_targets
            else:
                skipped_optimizer_steps += 1
        history.append(loss_value)
        duration_seconds = monotonic() - started_at
        timed_out = max_seconds is not None and duration_seconds >= max_seconds
        if timed_out and step < args.steps:
            stopped_reason = "time_budget"
        finishing = timed_out or step == args.steps
        if step == 1 or step % args.log_every == 0 or finishing:
            print(f"step={step} loss={loss_value:.4f} device={device}")
        try:
            if validation_encoded is not None and (
                step % args.eval_every == 0 or finishing
            ):
                validation = _validation_summary(
                    torch, model, validation_encoded, args.batch_size, device, aux_loss_weight
                )
                validation["step"] = step
                validation_history.append(validation)
                accuracy = validation["task_accuracy"]
                print(
                    f"eval_step={step} val_loss={validation['loss']:.4f} "
                    f"task_accuracy={'n/a' if accuracy is None else format(accuracy, '.3f')}"
                )
                lm_loss = validation["lm_loss"]
                if best_path is not None and math.isfinite(lm_loss) and (best_loss is None or lm_loss < best_loss):
                    save_best(step, validation)
                    # Set only after the write, so the full checkpoint never records a best file that failed.
                    best_loss, best_step = lm_loss, step
                model.train()
        finally:
            # Preserve completed updates even if final validation fails.
            if step % save_every == 0 or finishing:
                duration_seconds = monotonic() - started_at
                save(step, loss_value)
        if timed_out:
            break

    if successful_optimizer_steps == 0:
        raise RuntimeError(
            "training completed no optimizer updates; checkpoint saved for diagnosis. "
            "Reduce the learning rate or use --precision fp32"
        )
    final_loss = history[-1]
    return {
        "records": len(examples),
        "objective": objective,
        "packed_rows": len(encoded),
        "aux_loss_weight": aux_loss_weight,
        "steps": step,
        "requested_steps": args.steps,
        "completed_steps": step - start_step,
        "stopped_reason": stopped_reason,
        "duration_seconds": duration_seconds,
        "fused_adamw": fused_adamw,
        "processed_input_tokens": processed_input_tokens,
        "supervised_tokens": supervised_tokens,
        "updated_supervised_tokens": updated_supervised_tokens,
        "supervised_tokens_per_second": supervised_tokens / max(duration_seconds, 1e-9),
        "throughput_scope": "current invocation; includes validation and earlier saves, excludes final checkpoint write",
        "max_seconds": max_seconds,
        "final_loss": final_loss,
        "checkpoint": str(output_path),
        "device": str(device),
        "resumed_from_step": start_step,
        "validation": validation_history[-1] if validation_history else None,
        "best_step": best_step,
        "best_eval_lm_loss": best_loss,
        "best_out": str(best_path) if best_path else None,
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Supervised JSONL records (prompt-masked loss plus auxiliary heads)")
    source.add_argument(
        "--pretrain-text",
        help="Raw pretraining text: a .jsonl file with a \"text\" field per line, or any other file read "
             "as plain text and split into documents at form feeds and at blank lines (a newline followed "
             "by one or more whitespace-only lines); a file with neither is one document. "
             "Each document gets BOS/EOS, and documents are packed into block_size rows, trained with "
             "next-token loss on every position, and skip the task/error/confidence losses",
    )
    parser.add_argument("--out", default="artifacts/demo.pt")
    parser.add_argument(
        "--best-out",
        help="With --eval-data or --pretrain-eval-text, also save the weights from the validation with the "
             "lowest held-out lm_loss so far to this model-only checkpoint; a resume to the same file keeps "
             "a better earlier best",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--max-seconds", type=float,
        help="Stop at a completed training step after this many seconds; reserve extra time for validation and saving",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--architecture", choices=("modern", "legacy"))
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--learning-rate", type=float,
        help=f"Peak learning rate (default {DEFAULT_LEARNING_RATE:g}). With --resume it defaults to "
             "the checkpoint's rate, and a different value is rejected unless --override-learning-rate is set",
    )
    parser.add_argument(
        "--override-learning-rate", action="store_true",
        help="With --resume, replace the checkpoint's peak learning rate by --learning-rate while keeping "
             "the schedule position and optimizer moments",
    )
    parser.add_argument(
        "--weight-decay", type=float,
        help=f"AdamW weight decay (default {DEFAULT_WEIGHT_DECAY:g}, or the checkpoint's decay on --resume)",
    )
    parser.add_argument(
        "--warmup-steps", type=int,
        help=f"Learning-rate warmup steps (default {DEFAULT_WARMUP_STEPS}, or the checkpoint's warmup on --resume)",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-data", help="Held-out supervised JSONL, usable with --data or --pretrain-text")
    parser.add_argument(
        "--pretrain-eval-text",
        help="Held-out raw text in --pretrain-text format, packed the same way; excludes --eval-data",
    )
    parser.add_argument(
        "--aux-loss-weight", type=float,
        help=f"Weight of each task/error/confidence loss (default {DEFAULT_AUX_LOSS_WEIGHT:g}, or the "
             "checkpoint's weight on --resume); "
             "set 0 for SFT data whose labels are constant placeholders",
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-data-change", action="store_true",
        help="With --resume, accept a checkpoint recorded with a different training file, objective, or "
             "block size (for example SFT from pretrained weights); the sample stream restarts at 0",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preset", choices=tuple(MODEL_PRESETS), default="demo")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true", help="Use fused CUDA AdamW while retaining checkpoint optimizer moments")
    parser.add_argument("--save-every", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name, value in MODEL_PRESETS[args.preset].items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.warmup_steps is None and not args.resume:
        args.warmup_steps = DEFAULT_WARMUP_STEPS
    try:
        _runtime_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.n_layer < 1 or args.n_head < 1 or args.n_embd < 1:
        parser.error("--n-layer, --n-head, and --n-embd must be positive")
    # nan passes a plain "< 0" check, and nan or inf in any of these makes the weights non-finite after one update.
    if (args.weight_decay is not None and not 0 <= args.weight_decay < math.inf) or (args.warmup_steps or 0) < 0 \
            or args.eval_every < 1:
        parser.error("--weight-decay must be non-negative and finite; --warmup-steps and --eval-every must be positive")
    if (args.learning_rate is not None and not 0 < args.learning_rate < math.inf) or not 0 < args.grad_clip < math.inf \
            or args.log_every < 1:
        parser.error("--learning-rate and --grad-clip must be positive and finite; --log-every must be positive")
    if (args.warmup_steps or 0) > args.steps:
        parser.error("--warmup-steps cannot exceed --steps")
    if args.dry_run and args.resume:
        parser.error("--dry-run cannot be combined with --resume")
    if args.override_learning_rate and not (args.resume and args.learning_rate is not None):
        parser.error("--override-learning-rate requires --resume and --learning-rate")
    if args.eval_data and args.pretrain_eval_text:
        parser.error("--eval-data cannot be combined with --pretrain-eval-text")
    if args.allow_data_change and not args.resume:
        parser.error("--allow-data-change requires --resume")
    train_path = Path(args.data or args.pretrain_text).resolve()
    for flag, path in (("--eval-data", args.eval_data), ("--pretrain-eval-text", args.pretrain_eval_text)):
        if path and Path(path).resolve() == train_path:
            parser.error(f"{flag} must be different from the training data")
    out_path = Path(args.out).resolve()
    for flag in ("--data", "--pretrain-text", "--eval-data", "--pretrain-eval-text"):
        path = getattr(args, flag[2:].replace("-", "_"))
        if path and Path(path).resolve() == out_path:
            parser.error(f"--out must not be the {flag} file, or the first checkpoint save would replace that data")
    if args.best_out:
        if not (args.eval_data or args.pretrain_eval_text):
            parser.error("--best-out requires --eval-data or --pretrain-eval-text")
        best_path = Path(args.best_out).resolve()
        # The weights-only best file would also leave a --resume file unable to restore its optimizer.
        for flag in ("--out", "--resume", "--data", "--pretrain-text", "--eval-data", "--pretrain-eval-text"):
            path = getattr(args, flag[2:].replace("-", "_"))
            if path and Path(path).resolve() == best_path:
                parser.error(f"--best-out must not be the {flag} file, or saving the best weights would replace it")
    try:
        result = train(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
