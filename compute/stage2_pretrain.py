"""Pretrain slm-160m from scratch on the stage 1 corpus, one Kaggle session at a time.

Session 1 starts fresh with --preset slm-160m. Session k > 1 attaches session k-1's output and resumes
its optimizer, scheduler and step, so the cosine schedule spans all sessions (--steps is
PRETRAIN_TOTAL_STEPS). The trainer rebuilds the cosine from --steps on every resume, so lowering the
total before a session shortens the decay that is left; sessions 1 to 4 ran with 22,889. Each session
trains on its own byte range of pretrain_train.jsonl because cognition_slm.train packs the whole file
into Python lists, and the full 1.5 GB corpus would need roughly 25 GB of host RAM against Kaggle's
~29 GB. Session k's shard starts at start_step * TOKENS_PER_STEP byte tokens. The trainer samples
each shard in shuffled order and a session usually stops before finishing its oversized shard, so
consecutive shards overlap: some documents near a shard boundary are read twice and some are never
read. The token count trained on still matches the steps.

GPU memory on a 16 GB T4 (an estimate, checked only when session 1 runs): 160.7M fp32 weights,
gradients and two AdamW moments take 16 bytes per parameter, about 2.6 GB. With gradient
checkpointing the saved residual stream is 8 x 2048 x 768 x 4 bytes x 17 layers, about 0.9 GB, and
recomputing one block in backward needs its SwiGLU (3 x 8 x 2048 x 3072 x 2 bytes, about 0.3 GB) and
attention. SDPA on sm75 falls back to the math kernel when it cannot use a masked efficient kernel,
which materializes 8 x 12 x 2048 x 2048 fp16 scores (about 0.8 GB per copy, a few copies in backward).
The byte vocabulary keeps logits at 8 x 2048 x 259 floats. That totals roughly 6 to 8 GB, so batch 8
with accumulation 4 (65,536 tokens per optimizer step) leaves headroom.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INPUT = Path("/kaggle/input")
CHECKPOINT_NAME = "slm-160m-pretrain.pt"
KAGGLE_SESSION_LIMIT_SECONDS = 12 * 3600
# Covers shard selection, packing inside the trainer, the final validation and the last save.
SETUP_AND_SAVE_RESERVE_SECONDS = 1800
# About 1,000 packed rows; the full ~7.5 MB held-out split would add minutes to every evaluation.
PRETRAIN_EVAL_TOKENS = 2_000_000
LEARNING_RATE = "0.0003"
WARMUP_STEPS = 500
# Session 1 has no measured speed. Sizing its shard for a 40% faster step than the estimate keeps a
# quick T4 from wrapping around its shard, at the cost of some unread rows if the estimate holds.
UNMEASURED_SPEEDUP = 0.6
MEASURED_SPEEDUP = 0.9
SEED = 160


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def find_one(name: str, root: Path = INPUT) -> Path:
    matches = sorted(root.rglob(name)) if root.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} under {root}, found {len(matches)}: "
                           f"{[str(path) for path in matches][:5]}; check the kernel's attached inputs")
    return matches[0]


def document_tokens(line: str) -> int:
    """Byte tokens a JSONL document adds to the packed stream: UTF-8 bytes plus BOS and EOS."""
    return len(json.loads(line)["text"].encode("utf-8")) + 2


def select_shard_indices(sizes: list[int], start: int, length: int) -> list[int]:
    """Documents starting at or after token offset `start` until `length` tokens, wrapping once.

    Wrapping only matters when a run outlasts one pass; it stops before the first selected document,
    so a shard never holds the same document twice.
    """
    total = sum(sizes)
    if total <= 0 or length <= 0:
        raise ValueError(f"need a non-empty corpus and positive length, got {total} tokens and {length}")
    start %= total
    offsets, offset = [], 0
    for size in sizes:
        offsets.append(offset)
        offset += size
    first = next((index for index, value in enumerate(offsets) if value >= start), len(sizes))
    order = list(range(first, len(sizes))) + list(range(first))
    selected, taken = [], 0
    for index in order:
        if taken >= length:
            break
        selected.append(index)
        taken += sizes[index]
    return selected


def write_shard(source: Path, destination: Path, start: int, length: int) -> dict:
    with source.open(encoding="utf-8") as stream:
        sizes = [document_tokens(line) for line in stream if line.strip()]
    chosen = set(select_shard_indices(sizes, start, length))
    documents = tokens = 0
    with source.open(encoding="utf-8") as stream, destination.open("w", encoding="utf-8") as out:
        for index, line in enumerate(line for line in stream if line.strip()):
            if index in chosen:
                out.write(line if line.endswith("\n") else line + "\n")
                documents += 1
                tokens += sizes[index]
    return {"source": str(source), "start_token": start % sum(sizes), "requested_tokens": length,
            "documents": documents, "tokens": tokens, "corpus_tokens": sum(sizes),
            "corpus_documents": len(sizes), "sha256": digest(destination)}


def plan_session(start_step: int, total_steps: int, tokens_per_step: int,
                 session_seconds: float, seconds_per_step: float, measured: bool) -> dict:
    """Shard size for one session: as many steps as a faster-than-expected GPU could finish."""
    from compute.stages import planned_steps

    remaining = total_steps - start_step
    if remaining <= 0:
        raise ValueError(f"pretraining already reached step {start_step} of {total_steps}; run the sft stage")
    speed = seconds_per_step * (MEASURED_SPEEDUP if measured else UNMEASURED_SPEEDUP)
    shard_steps = max(1, min(remaining, planned_steps(session_seconds, speed, 0)))
    return {"start_step": start_step, "remaining_steps": remaining, "shard_steps": shard_steps,
            "start_token": start_step * tokens_per_step, "shard_tokens": shard_steps * tokens_per_step,
            "seconds_per_step_assumed": seconds_per_step, "seconds_per_step_measured": measured}


def pretrain_arguments(shard: Path, eval_subset: Path, output: Path, budget: float,
                       resume: Path | None) -> list[str]:
    """cognition_slm.train arguments; the peak rate and preset come from the checkpoint on resume."""
    from compute.stages import PRETRAIN_BATCH_SIZE, PRETRAIN_GRADIENT_ACCUMULATION, PRETRAIN_TOTAL_STEPS

    arguments = ["--pretrain-text", str(shard), "--pretrain-eval-text", str(eval_subset), "--out", str(output),
                 "--steps", str(PRETRAIN_TOTAL_STEPS), "--max-seconds", str(int(budget)),
                 "--batch-size", str(PRETRAIN_BATCH_SIZE),
                 "--gradient-accumulation-steps", str(PRETRAIN_GRADIENT_ACCUMULATION),
                 "--precision", "fp16", "--device", "cuda", "--gradient-checkpointing", "--fused-adamw",
                 "--aux-loss-weight", "0", "--warmup-steps", str(WARMUP_STEPS),
                 "--save-every", "250", "--eval-every", "1000", "--log-every", "50", "--seed", str(SEED)]
    # Each session trains on a fresh shard, so the resume data fingerprint differs by design.
    return arguments + (["--resume", str(resume), "--allow-data-change"] if resume
                        else ["--preset", "slm-160m", "--learning-rate", LEARNING_RATE])


def final_training_report(stdout: str) -> dict:
    """The JSON object cognition_slm.train prints last, which starts on a line holding only '{'."""
    lines = stdout.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "{"]
    if not starts:
        raise RuntimeError("training output did not contain a final JSON report")
    # stderr is merged into stdout, so an interpreter-exit warning may follow the report.
    report, _ = json.JSONDecoder().raw_decode("\n".join(lines[starts[-1]:]))
    return report


def bits_per_byte(nats_per_token: float | None) -> float | None:
    # Byte tokens make this bits per byte, up to the BOS/EOS tokens counted as targets.
    return None if nats_per_token is None else nats_per_token / math.log(2)


def measured_seconds_per_step(report: dict) -> float | None:
    steps, seconds = report.get("completed_steps"), report.get("duration_seconds")
    if not isinstance(steps, int) or steps <= 0 or not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    return seconds / steps


def run_logged(command: list[str], log_path: Path) -> str:
    # Stream the child log so a crash or session kill still leaves its traceback on disk and stdout.
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Training exited with {process.returncode}; see {log_path}")
    return log_path.read_text()


def setup_kaggle(label: str, require_gpu: bool = True):
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError(f"{label} must run on Kaggle; local model execution is prohibited")
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
    os.environ.update(PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    import torch

    gpu = torch.cuda.is_available() and (torch.ones(1, device="cuda") + 1).item() == 2
    if require_gpu and not gpu:
        raise RuntimeError("Enable a working NvidiaTeslaT4 GPU (kaggle kernels push --accelerator NvidiaTeslaT4)")
    # On a CPU session the matmuls run on the host, so they get every core instead of the two left for data.
    torch.set_num_threads(2 if gpu else os.cpu_count() or 2)
    manifest_path = ROOT / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, expected in manifest.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}; rebuild the bundle with compute/package.py")
    return torch, manifest


def previous_session_speed(session: int) -> float | None:
    reports = list(INPUT.rglob(f"pretrain_session_{session - 1}.json")) if INPUT.is_dir() else []
    if len(reports) != 1:
        return None
    return json.loads(reports[0].read_text()).get("seconds_per_step")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=int, required=True, help="1 for a fresh start, k to resume session k-1")
    args = parser.parse_args(argv)
    if args.session < 1:
        parser.error("--session must be at least 1")
    started = time.monotonic()
    torch, manifest = setup_kaggle("Pretraining")
    from compute.stages import (PRETRAIN_SESSION_SECONDS, PRETRAIN_TOTAL_STEPS, SECONDS_PER_STEP_ESTIMATE,
                                SESSION_RESERVE_SECONDS, TOKENS_PER_STEP, planned_steps)
    from cognition_slm.config import MODEL_PRESETS

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report_path = ROOT / f"pretrain_session_{args.session}.json"
    train_source, eval_source = find_one("pretrain_train.jsonl"), find_one("pretrain_eval.jsonl")
    report = {"status": "preparing", "session": args.session, "source_manifest": manifest,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "total_steps": PRETRAIN_TOTAL_STEPS, "tokens_per_step": TOKENS_PER_STEP}
    write_json(report_path, report)

    start_step, resume = 0, None
    if args.session > 1:
        previous = find_one(CHECKPOINT_NAME)
        payload = torch.load(previous, map_location="cpu", weights_only=True)
        preset = MODEL_PRESETS["slm-160m"]
        if any(payload["model_config"].get(key) != value for key, value in preset.items()):
            raise RuntimeError(f"{previous} is not an slm-160m checkpoint: {payload['model_config']}")
        start_step = int(payload["metadata"]["step"])
        # This session reads a new shard, so its permutation starts at 0 instead of mid-epoch of the old one.
        payload["metadata"] = {**payload["metadata"], "samples_seen": 0,
                               "previous_checkpoint_sha256": digest(previous)}
        resume = artifacts / "pretrain_resume.pt"
        torch.save(payload, resume)
        del payload
        report["previous_checkpoint"] = str(previous)
    measured = previous_session_speed(args.session)
    speed = measured or SECONDS_PER_STEP_ESTIMATE
    budget = min(PRETRAIN_SESSION_SECONDS,
                 KAGGLE_SESSION_LIMIT_SECONDS - (time.monotonic() - started) - SETUP_AND_SAVE_RESERVE_SECONDS)
    plan = plan_session(start_step, PRETRAIN_TOTAL_STEPS, TOKENS_PER_STEP, budget, speed, measured is not None)
    plan["expected_steps_this_session"] = min(plan["remaining_steps"],
                                              planned_steps(budget, speed, SESSION_RESERVE_SECONDS))
    shard, eval_subset = artifacts / "pretrain_shard.jsonl", artifacts / "pretrain_eval_subset.jsonl"
    plan["shard"] = write_shard(train_source, shard, plan["start_token"], plan["shard_tokens"])
    plan["eval_subset"] = write_shard(eval_source, eval_subset, 0, PRETRAIN_EVAL_TOKENS)
    budget = min(budget, KAGGLE_SESSION_LIMIT_SECONDS - (time.monotonic() - started) - SETUP_AND_SAVE_RESERVE_SECONDS)
    if budget < 600:
        raise RuntimeError(f"Only {budget:.0f} s of the Kaggle session remain after setup; rerun session {args.session}")
    report.update(status="training", plan=plan, max_seconds=budget)
    write_json(report_path, report)

    output = artifacts / CHECKPOINT_NAME
    command = [sys.executable, "-m", "cognition_slm.train",
               *pretrain_arguments(shard, eval_subset, output, budget, resume)]
    training = final_training_report(run_logged(command, ROOT / f"pretrain_session_{args.session}.log"))
    for temporary in (shard, resume):
        if temporary:
            temporary.unlink(missing_ok=True)
    validation = training.get("validation") or {}
    step = int(training["steps"])
    report.update(
        status="complete" if step >= PRETRAIN_TOTAL_STEPS else "session_complete_resume_next",
        training=training, step_reached=step, completed_steps=training["completed_steps"],
        seconds_per_step=measured_seconds_per_step(training),
        seconds_per_step_scope="training wall time / completed steps, including periodic eval and saves",
        eval_bits_per_byte=bits_per_byte(validation.get("lm_loss")),
        checkpoint=str(output), sha256=digest(output), elapsed_seconds=round(time.monotonic() - started, 1),
        next_session=None if step >= PRETRAIN_TOTAL_STEPS else args.session + 1)
    write_json(report_path, report)
    print("PRETRAIN_SESSION_COMPLETE", json.dumps({key: report[key] for key in
          ("session", "step_reached", "seconds_per_step", "eval_bits_per_byte", "next_session")}), flush=True)


if __name__ == "__main__":
    main()
