"""Instruction-tune the pretrained slm-160m on stage 3's short-answer English records.

The pretrain checkpoint sits at step PRETRAIN_TOTAL_STEPS with a fully annealed cosine schedule, so
resuming it as is would start SFT at a near-zero rate. The runner therefore writes an initialization
with the weights only: fresh AdamW moments, step 0 and a new warmup-plus-cosine schedule at 1e-4.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.stage2_pretrain import (INPUT, digest, final_training_report, find_one,  # noqa: E402
                                     run_logged, setup_kaggle, write_json)

PRETRAIN_NAME = "slm-160m-pretrain.pt"
OUTPUT_NAME = "slm-160m-sft.pt"
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
MIN_STEPS = 200
MAX_STEPS = 8000
MAX_SECONDS = 9 * 3600


def sft_steps(records: int, effective_batch: int = BATCH_SIZE * GRADIENT_ACCUMULATION, epochs: float = EPOCHS,
              minimum: int = MIN_STEPS, maximum: int = MAX_STEPS) -> int:
    """Optimizer steps for `epochs` passes over the records, clamped to [minimum, maximum]."""
    if records <= 0 or effective_batch <= 0 or epochs <= 0:
        raise ValueError(f"records, effective_batch and epochs must be positive, got {records}, "
                         f"{effective_batch}, {epochs}")
    return max(minimum, min(maximum, math.ceil(records * epochs / effective_batch)))


def sft_arguments(initialization: Path, train_path: Path, eval_path: Path, output: Path, steps: int) -> list[str]:
    return ["--resume", str(initialization), "--data", str(train_path), "--eval-data", str(eval_path),
            "--out", str(output), "--steps", str(steps), "--max-seconds", str(MAX_SECONDS),
            "--learning-rate", "0.0001", "--override-learning-rate", "--aux-loss-weight", "0",
            "--batch-size", str(BATCH_SIZE), "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION),
            "--warmup-steps", str(min(100, steps // 10)), "--precision", "fp16", "--device", "cuda",
            "--gradient-checkpointing", "--fused-adamw",
            "--save-every", "250", "--eval-every", "250", "--log-every", "25", "--seed", "161"]


def session_number(checkpoint: Path) -> int | None:
    """Highest pretrain_session_<k>.json beside the checkpoint's project folder, if any."""
    numbers = [int(match.group(1)) for path in checkpoint.parent.parent.glob("pretrain_session_*.json")
               if (match := re.fullmatch(r"pretrain_session_(\d+)\.json", path.name))]
    return max(numbers) if numbers else None


def latest_checkpoint(name: str = PRETRAIN_NAME, root: Path = INPUT) -> Path:
    matches = sorted(root.rglob(name)) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    ranked = sorted((session_number(path) or -1, str(path)) for path in matches)
    if not ranked or ranked[-1][0] < 0 or (len(ranked) > 1 and ranked[-1][0] == ranked[-2][0]):
        raise RuntimeError(f"Cannot pick the latest {name} under {root} from {[path for _, path in ranked][:5]}; "
                           "attach only the last pretrain session")
    return Path(ranked[-1][1])


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def probes_in_training(train_path: Path) -> list[str]:
    # Probes that repeat a training prompt measure recall, not general English.
    from compute.stage5_evaluate import ENGLISH_PROBES

    prompts = {normalize(json.loads(line)["prompt"]) for line in train_path.read_text().splitlines() if line.strip()}
    return [identifier for identifier, prompt, _ in ENGLISH_PROBES if normalize(prompt) in prompts]


def holdout_prompts_in_training(train_path: Path, holdout_path: Path = ROOT / "data/simple_questions_holdout.json") -> list[str]:
    """Holdout ids whose prompt appears verbatim (up to case and spacing) in the SFT training split."""
    prompts = {normalize(json.loads(line)["prompt"]) for line in train_path.read_text().splitlines() if line.strip()}
    rows = json.loads(holdout_path.read_text())["rows"]
    return [row["id"] for row in rows if normalize(row["prompt"]) in prompts]


def merge_distill(train_path: Path, eval_path: Path, distill_paths: list[Path], output: Path) -> dict:
    """Append distilled rows to sft_train, skipping prompts already in train or in eval.

    Returns counts; with no distill file attached, output is not written and train_path stays in use.
    """
    from cognition_slm.audit import _prompt_key

    if len(distill_paths) > 1:
        raise RuntimeError(f"Expected at most one distill_train.jsonl, found {[str(path) for path in distill_paths]}")
    if not distill_paths:
        return {"attached": False, "added": 0}
    train_lines = [line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {_prompt_key(json.loads(line)["prompt"]) for line in train_lines}
    # Eval prompts must stay unseen, or the SFT eval loss would measure memorization.
    held_out = {_prompt_key(json.loads(line)["prompt"])
                for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    added, skipped = [], 0
    for line in distill_paths[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key = _prompt_key(json.loads(line)["prompt"])
        if key in seen or key in held_out:
            skipped += 1
            continue
        seen.add(key)
        added.append(line)
    output.write_text("\n".join(train_lines + added) + "\n", encoding="utf-8")
    return {"attached": True, "path": str(distill_paths[0]), "added": len(added), "skipped_duplicate": skipped}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    args = parser.parse_args(argv)
    started = time.monotonic()
    torch, manifest = setup_kaggle("SFT")
    from cognition_slm.checkpoint import load_checkpoint_payload
    from cognition_slm.data import load_jsonl

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report_path = ROOT / "sft_report.json"
    source = latest_checkpoint()
    train_path, eval_path = find_one("sft_train.jsonl"), find_one("sft_eval.jsonl")
    merged = artifacts / "sft_train_with_distill.jsonl"
    distill = merge_distill(train_path, eval_path, sorted(INPUT.rglob("distill_train.jsonl")), merged)
    if distill["attached"]:
        train_path = merged
    # load_jsonl validates every record, so a malformed row fails here instead of mid-training.
    records, held_out = len(load_jsonl(train_path)), len(load_jsonl(eval_path))
    # Stage 3 filters these; checking again before GPU hours keeps the stage 5 holdout score honest.
    leaked = holdout_prompts_in_training(train_path)
    if leaked:
        raise RuntimeError(f"sft_train.jsonl contains holdout prompts {leaked}; rebuild it with compute/stage3_sft_data.py")
    steps = sft_steps(records, epochs=args.epochs)
    payload, config = load_checkpoint_payload(torch, source, inference_only=True)
    parent = payload.get("metadata", {})
    initialization = artifacts / "sft_initialization.pt"
    torch.save({"model_config": payload["model_config"], "model_state_dict": payload["model_state_dict"],
                "metadata": {"step": 0, "initialization_source": str(source),
                             "parent_step": parent.get("step"),
                             "optimizer_reset_reason": "new SFT data and learning-rate schedule"}}, initialization)
    del payload
    sessions = session_number(source)
    parent_report = source.parent.parent / f"pretrain_session_{sessions}.json" if sessions else None
    report = {"status": "training", "source_manifest": manifest, "gpu": torch.cuda.get_device_name(0),
              "pretrain_checkpoint": str(source), "pretrain_sha256": digest(source),
              "pretrain_step": parent.get("step"), "pretrain_session": sessions,
              "pretrain_status": json.loads(parent_report.read_text()).get("status")
              if parent_report and parent_report.exists() else None,
              "block_size": config.block_size, "train_records": records, "eval_records": held_out,
              "data_hashes": {"sft_train": digest(train_path), "sft_eval": digest(eval_path)},
              "probes_in_training": probes_in_training(train_path), "steps": steps, "epochs": args.epochs,
              "distill": distill}
    manifests = list(INPUT.rglob("sft_manifest.json"))
    if len(manifests) == 1:
        report["sft_manifest"] = json.loads(manifests[0].read_text())
    write_json(report_path, report)

    output = artifacts / OUTPUT_NAME
    command = [sys.executable, "-m", "cognition_slm.train",
               *sft_arguments(initialization, train_path, eval_path, output, steps)]
    training = final_training_report(run_logged(command, ROOT / "sft_training.log"))
    initialization.unlink()
    # Studio only needs weights; dropping optimizer state cuts the download from about 1.9 GB to 0.6 GB.
    compact, _ = load_checkpoint_payload(torch, output, inference_only=True)
    temporary = output.with_suffix(".compact.pt")
    torch.save(compact, temporary)
    del compact
    temporary.replace(output)
    validation = training.get("validation") or {}
    report.update(status="complete", training=training, checkpoint=str(output), sha256=digest(output),
                  checkpoint_format="model_only", eval_lm_loss=validation.get("lm_loss"),
                  elapsed_seconds=round(time.monotonic() - started, 1))
    write_json(report_path, report)
    print("SFT_COMPLETE", json.dumps({"steps": training["steps"], "eval_lm_loss": report["eval_lm_loss"],
                                      "sha256": report["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
