"""Compare saved model checkpoints on the same held-out coding set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint_payload
from .data import load_jsonl
from .evaluate import evaluate
from .generate import load_checkpoint


SCALAR_METRICS = (
    "task_accuracy",
    "error_accuracy",
    "confidence_bucket_accuracy",
    "exact_match_accuracy",
)


def parse_model_spec(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("model must use LABEL=CHECKPOINT format")
    return label.strip(), Path(path.strip())


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _scalar_metrics(result: dict) -> dict[str, float]:
    metrics = {name: float(result[name]) for name in SCALAR_METRICS}
    code_metrics = result.get("code_metrics")
    if code_metrics is not None:
        for name in ("syntax_validity", "required_symbol_recall", "static_score"):
            metrics[f"code_{name}"] = float(code_metrics[name])
    return metrics


def benchmark(checkpoints: list[tuple[str, Path]], data_path: str | Path, max_new_tokens: int) -> dict:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    examples = load_jsonl(data_path)
    models = []
    for label, path in checkpoints:
        model, tokenizer = load_checkpoint(path, torch.device("cpu"))
        result = evaluate(model, tokenizer, examples, max_new_tokens=max_new_tokens)
        models.append(
            {
                "label": label,
                "checkpoint": str(path),
                "architecture": model.config.architecture,
                "parameters": parameter_count(model),
                "metrics": _scalar_metrics(result),
            }
        )
    reference = models[0]
    deltas = []
    for item in models[1:]:
        deltas.append(
            {
                "label": item["label"],
                "reference": reference["label"],
                "metrics": {
                    name: item["metrics"].get(name, 0.0) - reference["metrics"].get(name, 0.0)
                    for name in sorted(set(reference["metrics"]).union(item["metrics"]))
                },
            }
        )
    return {
        "data": str(data_path),
        "records": len(examples),
        "max_new_tokens": max_new_tokens,
        "reference": reference["label"],
        "models": models,
        "deltas": deltas,
    }


def _device_checkpoint_check(path: Path) -> None:
    checkpoint, _ = load_checkpoint_payload(torch, path)
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"checkpoint {path} has invalid model_state_dict")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="LABEL=CHECKPOINT")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    try:
        checkpoints = [parse_model_spec(value) for value in args.model]
        for _, path in checkpoints:
            _device_checkpoint_check(path)
        result = benchmark(checkpoints, args.data, args.max_new_tokens)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
