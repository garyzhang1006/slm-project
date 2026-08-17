"""Evaluate held-out behavior labels and exact-match generations."""

from __future__ import annotations

import argparse
import json
import re

import torch

from .data import load_jsonl, format_prompt
from .generate import generate_text, load_checkpoint


def _normalize(text: str) -> str:
    text = re.sub(r"```(?:python)?", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().split())


@torch.no_grad()
def evaluate(model, tokenizer, examples, *, max_new_tokens: int) -> dict:
    model.eval()
    task_correct = 0
    error_correct = 0
    confidence_correct = 0
    exact_correct = 0
    rows = []
    for example in examples:
        prompt_ids = tokenizer.encode(format_prompt(example), add_eos=False)
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
        )
        output = model(input_ids, attention_mask=torch.ones_like(input_ids))
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
    return {
        "records": total,
        "task_accuracy": task_correct / total,
        "error_accuracy": error_correct / total,
        "confidence_bucket_accuracy": confidence_correct / total,
        "exact_match_accuracy": exact_correct / total,
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
