"""Evaluate held-out behavior labels and exact-match generations."""

from __future__ import annotations

import argparse
import json
import re

import torch

from .data import encode_prompt, load_jsonl
from .generate import generate_text, load_checkpoint


def _normalize(text: str) -> str:
    text = re.sub(r"```(?:python)?", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().split())


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int | None = None,
    calibration_bins: int = 10,
) -> dict[str, object]:
    """Return accuracy, macro-F1, ECE, and confusion data for one head."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (records, classes)")
    targets = targets.to(logits.device, dtype=torch.long).view(-1)
    if logits.size(0) == 0 or targets.numel() != logits.size(0):
        raise ValueError("logits and targets must contain the same non-zero record count")
    if num_classes is None:
        num_classes = logits.size(1)
    if num_classes != logits.size(1) or num_classes < 1:
        raise ValueError("num_classes must match logits and be positive")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")

    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    correct = predictions.eq(targets)
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=logits.device
    )
    confusion.index_put_((targets, predictions), torch.ones_like(targets), accumulate=True)
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1).float()
    recall = true_positive / confusion.sum(dim=1).clamp_min(1).float()
    f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).mean()

    confidence = probabilities.max(dim=-1).values
    ece = torch.zeros((), device=logits.device)
    for bin_index in range(calibration_bins):
        lower = bin_index / calibration_bins
        upper = (bin_index + 1) / calibration_bins
        in_bin = (confidence >= lower) & (
            (confidence < upper) if bin_index + 1 < calibration_bins else (confidence <= upper)
        )
        if torch.any(in_bin):
            weight = in_bin.float().mean()
            ece = ece + weight * (
                confidence[in_bin].mean() - correct[in_bin].float().mean()
            ).abs()
    return {
        "accuracy": float(correct.float().mean().item()),
        "macro_f1": float(f1.item()),
        "ece": float(ece.item()),
        "confusion_matrix": confusion.cpu().tolist(),
    }


@torch.no_grad()
def evaluate(model, tokenizer, examples, *, max_new_tokens: int) -> dict:
    model.eval()
    task_correct = 0
    error_correct = 0
    confidence_correct = 0
    exact_correct = 0
    task_logits = []
    task_targets = []
    error_logits = []
    error_targets = []
    confidence_logits = []
    confidence_targets = []
    rows = []
    for example in examples:
        prompt_ids = encode_prompt(example, tokenizer, model.config.block_size)
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
        )
        pool_positions = torch.tensor(
            [len(prompt_ids) - 1], dtype=torch.long, device=input_ids.device
        )
        output = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            pool_positions=pool_positions,
        )
        task_prediction = int(output.task_logits.argmax(dim=-1).item())
        error_prediction = int(output.error_logits.argmax(dim=-1).item())
        confidence_prediction = int(output.confidence_logits.argmax(dim=-1).item())
        generated = generate_text(
            model,
            tokenizer,
            example.prompt,
            task_type=example.task_type,
            max_new_tokens=max_new_tokens,
            temperature=0,
            top_k=0,
        )
        task_expected = model.config.task_types.index(example.task_type)
        error_expected = model.config.error_categories.index(example.error_category)
        task_logits.append(output.task_logits[0].detach().cpu())
        task_targets.append(task_expected)
        error_logits.append(output.error_logits[0].detach().cpu())
        error_targets.append(error_expected)
        confidence_logits.append(output.confidence_logits[0].detach().cpu())
        confidence_targets.append(example.confidence_bucket)
        task_correct += int(task_prediction == task_expected)
        error_correct += int(error_prediction == error_expected)
        confidence_correct += int(confidence_prediction == example.confidence_bucket)
        exact_correct += int(_normalize(generated) == _normalize(example.answer))
        rows.append(
            {
                "id": example.id,
                "generated": generated,
                "expected": example.answer,
                "task_prediction": model.config.task_types[task_prediction],
                "error_prediction": model.config.error_categories[error_prediction],
                "confidence_bucket_prediction": confidence_prediction,
            }
        )
    total = len(examples)
    task_metrics = classification_metrics(torch.stack(task_logits), torch.tensor(task_targets))
    error_metrics = classification_metrics(torch.stack(error_logits), torch.tensor(error_targets))
    confidence_metrics = classification_metrics(
        torch.stack(confidence_logits), torch.tensor(confidence_targets)
    )
    return {
        "records": total,
        "task_accuracy": task_correct / total,
        "error_accuracy": error_correct / total,
        "confidence_bucket_accuracy": confidence_correct / total,
        "exact_match_accuracy": exact_correct / total,
        "task_metrics": task_metrics,
        "error_metrics": error_metrics,
        "confidence_metrics": confidence_metrics,
        "rows": rows,
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = _device(args.device)
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    results = evaluate(model, tokenizer, load_jsonl(args.data), max_new_tokens=args.max_new_tokens)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
