"""Train the custom checkpoint on elementary QA and measure compositional transfer."""

import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PARENT_SHA = "ae2a1db39c1cd4722950b844d85a6cfd4e03add4a6c234f2488dc51abbbcc983"


def main():
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run elementary training on Kaggle; local model execution is prohibited")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts")]
    os.environ.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    from kaggle_english_run import digest, write_json
    from elementary_curriculum import build_curriculum
    from cognition_slm.data import load_jsonl
    from kaggle_qa_run import evaluate
    from kaggle_500m_quality_run import _final_training_report
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for elementary training")
    if shutil.disk_usage(root).free < 19 * 1024**3:
        raise RuntimeError("Need 19 GiB free for phase initialization and atomic checkpoint saves")
    torch.set_num_threads(2)
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    sources = list(Path("/kaggle/input").rglob("slm-500m-short-qa-pilot.pt"))
    if len(sources) != 1 or digest(sources[0]) != PARENT_SHA:
        raise RuntimeError("Attach the completed broad QA pilot checkpoint")
    source = sources[0]
    parent = json.loads((source.parent.parent / "short_qa_pilot_report.json").read_text())
    if parent.get("sha256") != PARENT_SHA or parent.get("status") != "complete_pending_manual_review":
        raise RuntimeError("Parent report and checkpoint do not match")
    replay_source = source.parent / "short_qa_train.jsonl"
    if digest(replay_source) != parent["selection"]["splits"]["train"]["sha256"]:
        raise RuntimeError("Parent QA replay corpus hash mismatch")
    audit = json.loads((root / "data/simple_questions_holdout.json").read_text())
    normalize = lambda text: " ".join(text.casefold().split())
    reserved = {normalize(row["prompt"]) for row in audit["rows"]}
    train, dev, curriculum = build_curriculum(reserved)
    excluded = reserved | {normalize(row["prompt"]) for row in dev}
    seen = {normalize(row["prompt"]) for row in train}
    replay = []
    for row in load_jsonl(replay_source):
        key = normalize(row.prompt)
        if key not in excluded | seen:
            replay.append(row.to_dict())
            seen.add(key)
        if len(replay) == 2000:
            break
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for name, rows in (("train", train + replay), ("dev", dev)):
        (artifacts / f"elementary_{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if payload["metadata"]["step"] != 1000 or payload["model_config"]["block_size"] != 2048:
        raise RuntimeError("Unexpected parent training state")
    # A new curriculum needs a fresh schedule; Adam moments and RNG remain intact.
    payload["metadata"] = {**payload["metadata"], "step": 0, "learning_rate": 0.00005,
                           "parent_cumulative_steps": parent["cumulative_steps"],
                           "schedule_restart_reason": "elementary QA curriculum"}
    payload.pop("scheduler_state_dict", None)
    for group in payload["optimizer_state_dict"]["param_groups"]:
        group.update(lr=0.00005, initial_lr=0.00005)
    initialization = artifacts / "elementary-init.pt"
    torch.save(payload, initialization)
    del payload
    gc.collect()
    items = [(row, [row["answer"]]) for row in dev[::max(1, len(dev) // 64)][:64]]
    report = {"status": "baseline", "parent_sha256": PARENT_SHA, "curriculum": curriculum,
              "parent_cumulative_steps": parent["cumulative_steps"],
              "replay_records": len(replay), "source_manifest": manifest,
              "evaluation_scope": "Template combination transfer; external audit requires manual review",
              "data_hashes": {name: digest(artifacts / f"elementary_{name}.jsonl") for name in ("train", "dev")}}
    destination = root / "elementary_report.json"
    write_json(destination, report)
    report["baseline"] = evaluate(torch, source, root, "elementary_baseline", items)
    report["status"] = "training"
    write_json(destination, report)
    output = artifacts / "slm-500m-elementary.pt"
    command = [sys.executable, "-m", "cognition_slm.train", "--resume", str(initialization),
               "--data", str(artifacts / "elementary_train.jsonl"),
               "--eval-data", str(artifacts / "elementary_dev.jsonl"), "--out", str(output),
               "--steps", "2000", "--max-seconds", "7200", "--warmup-steps", "100",
               "--learning-rate", "0.00005", "--batch-size", "1", "--gradient-accumulation-steps", "8",
               "--gradient-checkpointing", "--fused-adamw", "--precision", "fp16", "--device", "cuda",
               "--save-every", "500", "--eval-every", "500", "--log-every", "50", "--seed", "103"]
    log_path = root / "elementary_training.log"
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report["training"] = _final_training_report(log_path.read_text())
    report["phase_steps"] = report["training"]["completed_steps"]
    report["cumulative_steps"] = parent["cumulative_steps"] + report["phase_steps"]
    initialization.unlink()
    report.update(status="evaluating", sha256=digest(output), checkpoint=str(output))
    write_json(destination, report)
    report["final"] = evaluate(torch, output, root, "elementary_final", items)
    from cognition_slm.generate import load_checkpoint, generate_text
    model, tokenizer = load_checkpoint(output, torch.device("cuda"))
    report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    if report["parameters"] != 499_524_075:
        raise RuntimeError("Unexpected parameter count")
    report["simple_questions"] = []
    for row in audit["rows"]:
        answer = generate_text(model, tokenizer, row["prompt"], task_type=row["task_type"],
                               max_new_tokens=160, temperature=0, top_k=0)
        report["simple_questions"].append({**row, "answer": answer})
        write_json(destination, report)
    report["status"] = "complete_pending_manual_review"
    write_json(destination, report)


if __name__ == "__main__":
    main()
