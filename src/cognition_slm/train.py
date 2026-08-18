"""Train the tiny coding/cognition language model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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


def train(args: argparse.Namespace) -> dict:
    examples = load_jsonl(args.data)
    tokenizer = ByteTokenizer()
    config = ModelConfig(block_size=args.block_size)
    encoded = encode_examples(examples, tokenizer, config.block_size)
    if args.dry_run:
        return {
            "records": len(examples),
            "max_tokens": max(len(item["input_ids"]) for item in encoded),
            "block_size": config.block_size,
            "status": "dry-run",
        }

    torch = _import_torch()
    from .model import CognitionSLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(torch, args.device)
    model = CognitionSLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    history: list[float] = []
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        indices = [rng.randrange(len(encoded)) for _ in range(args.batch_size)]
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
        loss_value = float(output.loss.detach().cpu())
        history.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step={step} loss={loss_value:.4f} device={device}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metadata": {
                "records": len(examples),
                "steps": args.steps,
                "seed": args.seed,
                "device": str(device),
                "final_loss": history[-1],
            },
        },
        output_path,
    )
    return {
        "records": len(examples),
        "steps": args.steps,
        "final_loss": history[-1],
        "checkpoint": str(output_path),
        "device": str(device),
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
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
