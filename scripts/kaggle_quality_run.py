"""Train a 50M Studio checkpoint on Kaggle and probe English and Python."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    started = time.monotonic()
    from build_curriculum_data import build_rows, write_jsonl

    curriculum_train = root / "artifacts/curriculum_train.jsonl"
    curriculum_eval = root / "artifacts/curriculum_eval.jsonl"
    write_jsonl(build_rows("train"), curriculum_train)
    write_jsonl(build_rows("eval"), curriculum_eval)
    checkpoint = root / "artifacts/slm-50m-language-quality.pt"
    command = [
        sys.executable, "-m", "cognition_slm.train",
        "--data", str(curriculum_train),
        "--eval-data", str(curriculum_eval),
        "--out", str(checkpoint), "--preset", "slm-50m", "--steps", "1200",
        "--batch-size", "4", "--gradient-accumulation-steps", "2",
        "--gradient-checkpointing", "--precision", "fp16", "--device", "cuda",
        "--warmup-steps", "100", "--save-every", "200", "--eval-every", "300",
        "--log-every", "100", "--seed", "11",
    ]
    training = subprocess.run(command, check=True, capture_output=True, text=True)
    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    probes = {
        "hi": generate_text(
            model, tokenizer, "hi", task_type="language_generation",
            max_new_tokens=64, temperature=0, top_k=0,
        ),
        "grammar": generate_text(
            model, tokenizer, "Correct the grammar: She walk to school.", task_type="language_generation",
            max_new_tokens=48, temperature=0, top_k=0,
        ),
        "concept": generate_text(
            model, tokenizer, "Explain what a loop is in plain English.", task_type="language_generation",
            max_new_tokens=96, temperature=0, top_k=0,
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
            model, tokenizer,
            "Why use a set for membership checks?",
            task_type="code_explanation", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "algorithm": generate_text(
            model, tokenizer,
            "Describe the core idea behind binary search.",
            task_type="algorithm_reasoning", max_new_tokens=96, temperature=0, top_k=0,
        ),
        "capabilities": generate_text(
            model, tokenizer, "What can you help me with?", task_type="language_generation",
            max_new_tokens=96, temperature=0, top_k=0,
        ),
    }
    output_lines = training.stdout.splitlines()
    final_json_line = max(index for index, line in enumerate(output_lines) if line == "{")
    training_report = json.loads("\n".join(output_lines[final_json_line:]))
    report = {
        "kernel_purpose": "Studio quality checkpoint; project-authored synthetic English and Python curriculum",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "preset": "slm-50m",
        "checkpoint": str(checkpoint.relative_to(root)),
        "curriculum": {
            "train_records": len(build_rows("train")),
            "eval_records": len(build_rows("eval")),
            "source": "project-authored curriculum",
            "license": "CC0-1.0",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": training_report,
        "probes": probes,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (root / "quality_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("QUALITY_VERIFICATION_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
