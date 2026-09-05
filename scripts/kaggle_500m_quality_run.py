"""Train and probe the 500M Studio checkpoint on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _combined_curriculum(root: Path):
    from build_curriculum_data import build_rows, write_jsonl
    from cognition_slm.data import load_jsonl

    generated_train = build_rows("train")
    generated_eval = build_rows("eval")
    base_train = [example.to_dict() for example in load_jsonl(root / "data/demo.jsonl")]
    base_eval = [example.to_dict() for example in load_jsonl(root / "data/eval.jsonl")]
    train_rows = base_train + generated_train
    eval_rows = base_eval + generated_eval
    if len({row["id"] for row in train_rows}) != len(train_rows):
        raise RuntimeError("combined train curriculum contains duplicate ids")
    if len({row["id"] for row in eval_rows}) != len(eval_rows):
        raise RuntimeError("combined eval curriculum contains duplicate ids")
    train_path = root / "artifacts/curriculum_train_500m.jsonl"
    eval_path = root / "artifacts/curriculum_eval_500m.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(eval_rows, eval_path)
    return train_path, eval_path, {
        "base_train_records": len(base_train),
        "base_eval_records": len(base_eval),
        "generated_train_records": len(generated_train),
        "generated_eval_records": len(generated_eval),
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "source": "project-authored base data plus generated curriculum",
        "license": "CC0-1.0",
    }


def _final_training_report(stdout: str) -> dict:
    lines = stdout.splitlines()
    try:
        start = max(index for index, line in enumerate(lines) if line == "{")
    except ValueError as exc:
        raise RuntimeError("training output did not contain a final JSON report") from exc
    return json.loads("\n".join(lines[start:]))


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("This runner requires Kaggle; no local training is permitted")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    os.environ["PYTHONPATH"] = str(root / "src")

    import torch

    from cognition_slm.generate import generate_text, load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable; enable a GPU accelerator and rerun")
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {name}")

    started = time.monotonic()
    train_path, eval_path, curriculum = _combined_curriculum(root)
    subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train", str(train_path), "--eval", str(eval_path)], check=True)
    checkpoint = root / "artifacts/slm-500m-language-quality.pt"
    command = [
        sys.executable, "-m", "cognition_slm.train",
        "--data", str(train_path), "--eval-data", str(eval_path),
        "--out", str(checkpoint), "--preset", "slm-500m", "--steps", "600",
        "--batch-size", "1", "--gradient-accumulation-steps", "8",
        "--gradient-checkpointing", "--precision", "fp16", "--device", "cuda",
        "--warmup-steps", "50", "--save-every", "100", "--eval-every", "150",
        "--log-every", "50", "--seed", "17",
    ]
    training = subprocess.run(command, check=True, capture_output=True, text=True)
    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 499_524_075:
        raise RuntimeError(f"unexpected 500M parameter count: {parameters}")
    probes = {
        "hi": generate_text(
            model, tokenizer, "hi", task_type="language_generation",
            max_new_tokens=64, temperature=0, top_k=0,
        ),
        "grammar": generate_text(
            model, tokenizer, "Correct the grammar: She walk to school.",
            task_type="language_generation", max_new_tokens=48, temperature=0, top_k=0,
        ),
        "concept": generate_text(
            model, tokenizer, "Explain what a loop is in plain English.",
            task_type="language_generation", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "code_square": generate_text(
            model, tokenizer,
            "Write a Python function named square that returns the square of a number.",
            task_type="code_generation", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "code_dedupe": generate_text(
            model, tokenizer,
            "Write a Python function named dedupe that removes duplicates while preserving order.",
            task_type="code_generation", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "code_debugging": generate_text(
            model, tokenizer,
            "Fix this Python function: def add(left, right) return left + right",
            task_type="code_debugging", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "code_explanation": generate_text(
            model, tokenizer, "Why use a set for membership checks?",
            task_type="code_explanation", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "algorithm": generate_text(
            model, tokenizer, "Describe the core idea behind binary search.",
            task_type="algorithm_reasoning", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "capabilities": generate_text(
            model, tokenizer, "What can you help me with?", task_type="language_generation",
            max_new_tokens=96, temperature=0, top_k=0,
        ),
    }
    report = {
        "kernel_purpose": "500M Studio quality checkpoint; project-authored synthetic English and Python curriculum",
        "preset": "slm-500m",
        "context_window": model.config.block_size,
        "parameters": parameters,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint.relative_to(root)),
        "curriculum": curriculum,
        "training": _final_training_report(training.stdout),
        "probes": probes,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (root / "quality_verification_500m.json").write_text(json.dumps(report, indent=2) + "\n")
    print("QUALITY_VERIFICATION_500M_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
