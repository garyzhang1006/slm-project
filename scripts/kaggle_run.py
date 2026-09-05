"""Run the regression suite and a full-length 50M GPU smoke train on Kaggle."""

from __future__ import annotations

import hashlib
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
    from cognition_slm.config import MODEL_PRESETS, ModelConfig
    from cognition_slm.data import encode_examples, load_jsonl, validate_record
    from cognition_slm.generate import generate_text, load_checkpoint
    from cognition_slm.model import CognitionSLM
    from cognition_slm.tokenizer import ByteTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable; enable a GPU accelerator and rerun")
    torch.ones(1, device="cuda").add_(1)
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    started = time.monotonic()
    report = {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "source_manifest": manifest,
        "purpose": "pipeline verification only; synthetic smoke data cannot establish coding quality",
    }
    print(json.dumps({key: value for key, value in report.items() if key != "source_manifest"}), flush=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    report["regression_suite"] = "passed"
    subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train", "data/demo.jsonl", "--eval", "data/eval.jsonl"], check=True)

    # Exercise actual long sequences rather than merely raising the configured limit.
    long_data = root / "artifacts/long-context-smoke.jsonl"
    long_data.parent.mkdir(exist_ok=True)
    records = []
    for index in range(8):
        record = validate_record({
            "id": f"long-smoke-{index}",
            "prompt": "Read this module and implement identity.\n" + "# Project-authored test fixture.\n" * 51,
            "answer": "def identity(value):\n    return value\n" + "# Synthetic length coverage only.\n" * 18,
            "task_type": "code_generation", "confidence": 0.5,
            "error_category": "none", "source": "project-authored length smoke fixture",
            "license": "CC0-1.0",
        })
        records.append(record.to_dict())
    long_data.write_text("".join(json.dumps(record) + "\n" for record in records))
    encoded = encode_examples(load_jsonl(long_data), ByteTokenizer(), 2048)
    assert all(len(item["input_ids"]) == 2048 for item in encoded)
    report["actual_training_sequence_length"] = 2048
    config = ModelConfig(**MODEL_PRESETS["slm-50m"])
    with torch.device("meta"):
        model = CognitionSLM(config)
    report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    del model
    checkpoint = root / "artifacts/slm-50m-2048-smoke.pt"
    command = [sys.executable, "-m", "cognition_slm.train", "--data", str(long_data),
               "--out", str(checkpoint), "--preset", "slm-50m", "--steps", "8",
               "--batch-size", "1", "--gradient-accumulation-steps", "2",
               "--gradient-checkpointing", "--precision", "fp16", "--device", "cuda",
               "--warmup-steps", "2", "--save-every", "4", "--log-every", "1"]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-m", "cognition_slm.train", "--data", str(long_data),
                    "--resume", str(checkpoint), "--out", str(checkpoint), "--steps", "10",
                    "--batch-size", "1", "--gradient-accumulation-steps", "2",
                    "--gradient-checkpointing", "--precision", "fp16", "--device", "cuda",
                    "--warmup-steps", "2", "--log-every", "1"], check=True)
    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    assert model.config.block_size == 2048
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report["optimizer_steps"] = payload["metadata"]["optimizer_steps"]
    report["final_loss"] = payload["metadata"]["final_loss"]
    assert report["optimizer_steps"] > 0, "AMP skipped every optimizer update"
    assert all(torch.isfinite(value).all() for value in payload["model_state_dict"].values())
    del payload
    report["generated_smoke_text"] = generate_text(model, tokenizer, "Write identity(value).", max_new_tokens=16, temperature=0)
    report["checkpoint"] = str(checkpoint.relative_to(root))
    report["completed_steps"] = 10
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VERIFICATION_COMPLETE", json.dumps({key: value for key, value in report.items() if key != "source_manifest"}), flush=True)


if __name__ == "__main__":
    main()
