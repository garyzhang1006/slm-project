"""Train a usable 50M Studio checkpoint on Kaggle and probe basic prompts."""

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
    os.environ["PYTHONPATH"] = str(root / "src")

    import torch

    from cognition_slm.generate import generate_text, load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable; enable a GPU accelerator and rerun")
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    started = time.monotonic()
    checkpoint = root / "artifacts/slm-50m-language-quality.pt"
    command = [
        sys.executable, "-m", "cognition_slm.train",
        "--data", str(root / "data/demo.jsonl"),
        "--eval-data", str(root / "data/eval.jsonl"),
        "--out", str(checkpoint), "--preset", "slm-50m", "--steps", "600",
        "--batch-size", "4", "--gradient-accumulation-steps", "2",
        "--gradient-checkpointing", "--precision", "fp16", "--device", "cuda",
        "--warmup-steps", "50", "--save-every", "100", "--eval-every", "200",
        "--log-every", "50", "--seed", "11",
    ]
    training = subprocess.run(command, check=True, capture_output=True, text=True)
    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    probes = {
        "hi": generate_text(
            model, tokenizer, "hi", task_type="language_generation",
            max_new_tokens=64, temperature=0, top_k=0,
        ),
        "capabilities": generate_text(
            model, tokenizer, "What can you help me with?", task_type="language_generation",
            max_new_tokens=96, temperature=0, top_k=0,
        ),
        "code": generate_text(
            model, tokenizer,
            "Write a Python function that returns the square of a number.",
            task_type="code_generation", max_new_tokens=96, temperature=0, top_k=0,
        ),
    }
    output_lines = training.stdout.splitlines()
    final_json_line = max(index for index, line in enumerate(output_lines) if line == "{")
    training_report = json.loads("\n".join(output_lines[final_json_line:]))
    report = {
        "kernel_purpose": "Studio quality checkpoint; synthetic data is only a smoke corpus",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "preset": "slm-50m",
        "checkpoint": str(checkpoint.relative_to(root)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": training_report,
        "probes": probes,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (root / "quality_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("QUALITY_VERIFICATION_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
