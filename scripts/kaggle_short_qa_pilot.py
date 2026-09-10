"""Bounded short-answer learning diagnostic, exclusively on Kaggle."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PARENT_SHA256 = "df52511585d374a7ba2cc9f818e00c1ffad377bfb42bbe99aba6c873efa4e0ea"
CORPUS_SHA256 = "b9bd4d4bdac5624df44d8dbc94d4a7eccd6db85e7b564d0b1bad492f4a180203"


def prepare_phase_payload(payload: dict) -> dict:
    """Restart only the schedule; retain learned weights, Adam moments, scaler and RNG."""
    if payload["metadata"]["step"] != 6742:
        raise RuntimeError("Expected parent step 6742 before starting diagnostic phase")
    payload["metadata"] = {
        **payload["metadata"], "parent_step": 6742, "step": 0, "learning_rate": 0.00005,
        "scheduler_restart_reason": "Short-answer diagnostic phase needs a fresh finite learning-rate schedule",
    }
    payload.pop("scheduler_state_dict", None)
    for group in payload["optimizer_state_dict"]["param_groups"]:
        group["lr"] = 0.00005
        group["initial_lr"] = 0.00005
    return payload


def prepare_subset(source: Path, holdout: dict, artifacts: Path) -> dict:
    from cognition_slm.data import load_jsonl
    from kaggle_english_run import digest

    normalize = lambda text: " ".join(text.casefold().split())
    reserved = {normalize(row["prompt"]) for row in holdout["rows"]}
    unique = {}
    for example in load_jsonl(source):
        if (example.source != "databricks/databricks-dolly-15k"
                or example.task_type != "language_generation"
                or normalize(example.prompt) in reserved):
            continue
        row = example.to_dict()
        serialized = json.dumps(row, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        prompt_key = normalize(example.prompt)
        # Keep prompt duplicates in one partition, deterministically.
        if prompt_key not in unique or key < unique[prompt_key][0]:
            unique[prompt_key] = (key, row)
    ordered = [row for _, row in sorted(unique.values())]
    candidates = []
    for prompt_limit, answer_limit, minimum in ((160, 96, 320), (256, 128, 80)):
        candidates = [row for row in ordered
                      if len(row["prompt"].encode("utf-8")) <= prompt_limit
                      and len(row["answer"].encode("utf-8")) <= answer_limit]
        if len(candidates) >= minimum:
            break
    else:
        raise RuntimeError(f"Only {len(candidates)} eligible short Dolly rows; need at least 80")
    if prompt_limit == 160:
        splits = {"train": candidates[:256], "probe": candidates[256:320]}
    else:
        splits = {"train": candidates[:-16][:256], "probe": candidates[-16:]}
    result = {
        "source_sha256": CORPUS_SHA256, "eligible_count": len(candidates),
        "prompt_limit_utf8_bytes": prompt_limit, "answer_limit_utf8_bytes": answer_limit,
        "ordering": "SHA256 of canonical JSON row; normalized prompt deduplication",
        "probe_scope": "Excluded from this pilot only; historical training exposure. Not heldout generalization.",
        "splits": {},
    }
    for name, rows in splits.items():
        path = artifacts / f"short_qa_{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        result["splits"][name] = {
            "path": str(path), "count": len(rows), "sha256": digest(path),
            "ids": [row["id"] for row in rows],
            "prompt_bytes": [len(row["prompt"].encode("utf-8")) for row in rows],
            "answer_bytes": [len(row["answer"].encode("utf-8")) for row in rows],
            "sources": sorted({row["source"] for row in rows}),
            "licenses": sorted({row["license"] for row in rows}),
        }
    return result


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this pilot on Kaggle; local model execution is prohibited")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts")]
    os.environ.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    from kaggle_english_run import digest, write_json

    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    if shutil.disk_usage(root).free < 17 * 1024**3:
        raise RuntimeError("Checkpoint saves require at least 17 GiB of free working disk")
    matches = list(Path("/kaggle/input").rglob("slm-500m-efficient.pt"))
    if len(matches) != 1 or digest(matches[0]) != PARENT_SHA256:
        raise RuntimeError("Attach the completed efficient checkpoint with the recorded SHA256")
    source = matches[0]
    parent = json.loads((source.parent.parent / "efficient_report.json").read_text())
    if parent.get("status") != "complete_pending_answer_review" or parent.get("sha256") != PARENT_SHA256:
        raise RuntimeError("Parent report must identify the completed efficient checkpoint")
    corpus = source.parent / "broad_train.jsonl"
    if digest(corpus) != CORPUS_SHA256:
        raise RuntimeError("Attached broad training corpus does not match recorded SHA256")

    import torch
    from cognition_slm.checkpoint import load_checkpoint_payload
    from cognition_slm.data import load_jsonl
    from cognition_slm.generate import load_checkpoint, generate_text
    from kaggle_500m_quality_run import _final_training_report
    from kaggle_qa_run import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    torch.set_num_threads(2)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    required = ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                "torch_rng_state", "cuda_rng_state_all")
    for name in required:
        value = payload.get(name)
        if value is None or (value.numel() == 0 if isinstance(value, torch.Tensor) else not value):
            raise RuntimeError(f"Full continuation state required; missing or empty {name}")
    if payload["metadata"]["step"] != 6742 or payload["model_config"]["block_size"] != 2048:
        raise RuntimeError("Expected parent step 6742 and 2048-byte context")
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    holdout_path = root / "data/simple_questions_holdout.json"
    holdout = json.loads(holdout_path.read_text())
    selection = prepare_subset(corpus, holdout, artifacts)
    output = artifacts / "slm-500m-short-qa-pilot.pt"
    initialization = artifacts / "short-qa-phase-init.pt"
    torch.save(prepare_phase_payload(payload), initialization)
    del payload
    gc.collect()
    command = [sys.executable, "-m", "cognition_slm.train", "--resume", str(initialization),
               "--data", str(artifacts / "short_qa_train.jsonl"),
               "--eval-data", str(artifacts / "short_qa_probe.jsonl"),
               "--out", str(output), "--steps", "1000", "--max-seconds", "1800",
               "--learning-rate", "0.00005", "--warmup-steps", "50",
               "--batch-size", "1", "--gradient-accumulation-steps", "8",
               "--gradient-checkpointing", "--fused-adamw", "--precision", "fp16",
               "--device", "cuda", "--save-every", "250", "--eval-every", "500",
               "--log-every", "50", "--seed", "97"]
    report = {
        "status": "baseline", "parent_sha256": PARENT_SHA256, "source_manifest": manifest,
        "gpu": torch.cuda.get_device_name(0), "selection": selection, "command": command,
        "holdout_sha256": digest(holdout_path),
        "scope": "Diagnostic pilot only; no checkpoint promotion or automatic longer training",
        "schedule": "New 1000-step phase, 50-step warmup; preserve weights, Adam moments, scaler and RNG; reset scheduler",
        "parent_step": 6742,
    }
    destination = root / "short_qa_pilot_report.json"
    write_json(destination, report)
    evaluation = {
        name: [(row.to_dict(), [row.answer]) for row in load_jsonl(artifacts / f"short_qa_{name}.jsonl")[:16]]
        for name in ("train", "probe")
    }
    for name, items in evaluation.items():
        report[f"baseline_{name}"] = evaluate(torch, source, root, f"short_qa_baseline_{name}", items)
        write_json(destination, report)
    report["status"] = "training"
    write_json(destination, report)
    log_path = root / "short_qa_training.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError("Short QA pilot failed; inspect short_qa_training.log")
    report["training"] = _final_training_report(log_path.read_text())
    completed, _ = load_checkpoint_payload(torch, output, inference_only=True)
    report["phase_steps"] = int(completed["metadata"]["step"])
    report["cumulative_steps"] = 6742 + report["phase_steps"]
    del completed
    gc.collect()
    initialization.unlink()
    report.update(status="evaluating", checkpoint=str(output), sha256=digest(output))
    write_json(destination, report)
    for name, items in evaluation.items():
        report[f"final_{name}"] = evaluate(torch, output, root, f"short_qa_final_{name}", items)
        write_json(destination, report)
    model, tokenizer = load_checkpoint(output, torch.device("cuda"))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 499_524_075:
        raise RuntimeError(f"Unexpected model parameter count: {parameters}")
    report.update(parameters=parameters, simple_questions=[])
    for row in holdout["rows"]:
        answer = generate_text(model, tokenizer, row["prompt"], task_type=row["task_type"],
                               max_new_tokens=160, temperature=0, top_k=0)
        report["simple_questions"].append({**row, "answer": answer})
        write_json(destination, report)
    report["status"] = "complete_pending_manual_review"
    write_json(destination, report)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
