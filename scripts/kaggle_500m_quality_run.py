"""Train and probe the 500M Studio checkpoint on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


# Unseen wordings of the original probe skills; main() rejects any that match a training prompt.
QUALITY_PROBES = (
    ("hi", "hello there", "language_generation", 64),
    ("grammar", "Correct the grammar: He run to the store every morning.", "language_generation", 48),
    ("concept", "Explain what a tuple is in plain English.", "language_generation", 96),
    ("code_simple", "Write a Python function named halve that returns half of a number.", "code_generation", 96),
    ("code_dedupe", "Write a Python function named unique_words that returns the distinct words of a sentence in first-seen order.", "code_generation", 96),
    ("code_debugging", "Fix this Python function: def halve(value) return value / 2", "code_debugging", 96),
    ("code_explanation", "Why use a dictionary for lookups by key?", "code_explanation", 96),
    ("algorithm", "Describe the core idea behind merge sort.", "algorithm_reasoning", 96),
    ("capabilities", "What kinds of tasks can you do for me?", "language_generation", 96),
)


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
    # stderr is merged into the log, so an interpreter-exit warning may follow the report.
    report, _ = json.JSONDecoder().raw_decode("\n".join(lines[start:]))
    return report


def _probes_in_training(probes, train_path: Path) -> list[str]:
    # Probes that repeat a training prompt measure recall, not quality.
    train_prompts = {" ".join(json.loads(line)["prompt"].casefold().split())
                     for line in train_path.read_text().split("\n") if line.strip()}
    return [name for name, prompt, _, _ in probes if " ".join(prompt.casefold().split()) in train_prompts]


def _run_logged(command: list[str], log_path: Path) -> None:
    # Stream the child log so a crash or session kill still leaves its traceback on disk and stdout.
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Training failed; see {log_path.name}")


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
    leaked = _probes_in_training(QUALITY_PROBES, train_path)
    if leaked:
        raise RuntimeError(f"Quality probes duplicate training prompts: {leaked}")
    subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train", str(train_path), "--eval", str(eval_path)], check=True)
    checkpoint = root / "artifacts/slm-500m-language-quality.pt"
    command = [
        sys.executable, "-m", "cognition_slm.train",
        "--data", str(train_path), "--eval-data", str(eval_path),
        "--out", str(checkpoint), "--preset", "slm-500m", "--steps", "1200",
        "--batch-size", "1", "--gradient-accumulation-steps", "8",
        "--gradient-checkpointing", "--precision", "fp16", "--device", "cuda",
        "--warmup-steps", "100", "--save-every", "200", "--eval-every", "300",
        "--log-every", "100", "--seed", "17",
    ]
    log_path = root / "quality_500m_training.log"
    _run_logged(command, log_path)
    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 499_524_075:
        raise RuntimeError(f"unexpected 500M parameter count: {parameters}")
    probes = {
        name: generate_text(model, tokenizer, prompt, task_type=task,
                            max_new_tokens=max_new_tokens, temperature=0, top_k=0)
        for name, prompt, task, max_new_tokens in QUALITY_PROBES
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
        "training": _final_training_report(log_path.read_text()),
        "probes": probes,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (root / "quality_verification_500m.json").write_text(json.dumps(report, indent=2) + "\n")
    print("QUALITY_VERIFICATION_500M_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
