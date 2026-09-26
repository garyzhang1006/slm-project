"""Compare saved model checkpoints on the same held-out coding set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_jsonl
from .evaluate import _device, evaluate
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
    # task_accuracy is None when a legacy checkpoint lacks every eval task label.
    metrics = {name: float(result[name]) for name in SCALAR_METRICS if result[name] is not None}
    code_metrics = result.get("code_metrics")
    if code_metrics is not None:
        for name in ("syntax_validity", "required_symbol_recall", "static_score"):
            metrics[f"code_{name}"] = float(code_metrics[name])
    return metrics


def benchmark(
    checkpoints: list[tuple[str, Path]],
    data_path: str | Path,
    max_new_tokens: int,
    device: torch.device | None = None,
) -> dict:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    examples = load_jsonl(data_path)
    if device is None:
        device = torch.device("cpu")
    models = []
    for label, path in checkpoints:
        model, tokenizer = load_checkpoint(path, device)
        result = evaluate(model, tokenizer, examples, max_new_tokens=max_new_tokens)
        models.append(
            {
                "label": label,
                "checkpoint": str(path),
                "architecture": model.config.architecture,
                "parameters": parameter_count(model),
                "metrics": _scalar_metrics(result),
                "task_records": result["task_records"],
            }
        )
        del model, tokenizer
    reference = models[0]
    deltas = []
    for item in models[1:]:
        # Only metrics both models scored on the same records form a meaningful delta.
        names = set(reference["metrics"]).intersection(item["metrics"])
        if item["task_records"] != reference["task_records"]:
            names.discard("task_accuracy")
        deltas.append(
            {
                "label": item["label"],
                "reference": reference["label"],
                "metrics": {
                    name: item["metrics"][name] - reference["metrics"][name]
                    for name in sorted(names)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="LABEL=CHECKPOINT")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        checkpoints = [parse_model_spec(value) for value in args.model]
        result = benchmark(checkpoints, args.data, args.max_new_tokens, _device(args.device))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
