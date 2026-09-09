"""Continue the custom SLM on broader English and QA within a Kaggle time budget."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
PARENT_SHA256 = "99176c4bc495018f7cf27ed77559b05e709dc2b3d3d8243cf499d9f6ac6ae789"
DATA_HASHES = {
    "train": "e62e3d6f89ad4aff6180e9b58c8c4338caeb886c69b5b3662fa03d13c348bc14",
    "dev": "11c9eb1675128eb3d76ca735f58e208ee56f528c5f69db7703c4d265bbba3cd8",
    "test": "38f3de1c8a8cc959e7fde4788a96d86b4eff90663dacd8c54d6953ff775c2dc2",
}


def prepare_data(root: Path, parent: Path) -> dict:
    from datasets import load_dataset
    from cognition_slm.data import load_jsonl
    from build_curriculum_data import write_jsonl
    from broad_english_corpus import paragraph_records
    from english_corpus import deterministic_split, _normalized_prompt
    from kaggle_english_run import digest, write_json
    from kaggle_studio_verify import QUESTION_PROBES

    for split, expected in DATA_HASHES.items():
        source = parent / f"artifacts/qa_{split}.jsonl"
        if digest(source) != expected:
            raise RuntimeError(f"Parent QA {split} data hash mismatch")
        shutil.copyfile(source, root / f"artifacts/qa_{split}.jsonl")
    holdouts = load_jsonl(root / "artifacts/qa_dev.jsonl") + load_jsonl(root / "artifacts/qa_test.jsonl")
    excluded_passages = [" ".join(item.prompt.split("\n\nPassage: ", 1)[1]
                                  .rsplit("\n\nQuestion:", 1)[0].casefold().split()) for item in holdouts]
    reserved = {_normalized_prompt(item.prompt) for item in holdouts}
    reserved.update(_normalized_prompt(prompt) for _, prompt, _, _ in QUESTION_PROBES)
    train, evaluation = [], []
    seen_documents, seen_prompts = set(), set()
    scanned = 0
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          revision=FINEWEB_REVISION, split="train", streaming=True)
    for raw in stream:
        scanned += 1
        body = " ".join(str(raw.get("text", "")).casefold().split())
        key = hashlib.sha256(body.encode()).hexdigest()
        if key not in seen_documents and not any(passage in body for passage in excluded_passages):
            seen_documents.add(key)
            # A whole source document belongs to one split, including all its paragraphs.
            source_key = hashlib.sha256(str(raw.get("url", "")).encode()).hexdigest()
            target = evaluation if int(source_key, 16) % 50 == 0 else train
            for row in paragraph_records(raw):
                prompt_key = _normalized_prompt(row["prompt"])
                if prompt_key not in seen_prompts and prompt_key not in reserved:
                    seen_prompts.add(prompt_key)
                    target.append(row)
        if (len(train) >= 60_000 and len(evaluation) >= 128) or scanned >= 150_000:
            break
    if len(train) < 10_000 or len(evaluation) < 128:
        raise RuntimeError(f"Insufficient English paragraphs: {len(train)} train/{len(evaluation)} eval")
    evaluation = evaluation[:128]
    qa = [row.to_dict() for row in load_jsonl(root / "artifacts/qa_train.jsonl")]
    # Keep instruction practice during English training to reduce forgetting.
    mixed, _ = deterministic_split(train + qa, eval_count=0)
    write_jsonl(mixed, root / "artifacts/broad_train.jsonl")
    write_jsonl(evaluation, root / "artifacts/broad_eval.jsonl")
    for evaluation_path in ("broad_eval", "qa_dev", "qa_test"):
        subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train",
                        "artifacts/broad_train.jsonl", "--eval", f"artifacts/{evaluation_path}.jsonl"], check=True)
    report = {"fineweb_revision": FINEWEB_REVISION, "fineweb_subset": "sample-10BT",
              "documents_scanned": scanned, "english_train": len(train), "english_eval": len(evaluation),
              "qa_train": len(qa), "mixed_train": len(mixed), "hashes": {}}
    for name in ("broad_train", "broad_eval", "qa_train", "qa_dev", "qa_test"):
        report["hashes"][name] = digest(root / f"artifacts/{name}.jsonl")
    write_json(root / "long_corpus_manifest.json", report)
    return report


def train_stage(torch, root: Path, source: Path, name: str, budget: float) -> tuple[Path, dict]:
    from kaggle_english_run import initialize_stage
    from kaggle_500m_quality_run import _final_training_report

    initialization = root / "artifacts/long_initialization.pt"
    initialize_stage(torch, source, initialization)
    output = root / f"artifacts/slm-500m-long-{name}.pt"
    train_name, eval_name = ("broad_train", "broad_eval") if name == "english" else ("qa_train", "qa_dev")
    command = [sys.executable, "-m", "cognition_slm.train", "--resume", str(initialization),
               "--data", f"artifacts/{train_name}.jsonl", "--eval-data", f"artifacts/{eval_name}.jsonl",
               "--out", str(output), "--steps", "100000", "--max-seconds", str(budget),
               "--learning-rate", "0.0001" if name == "english" else "0.00005",
               "--batch-size", "1", "--gradient-accumulation-steps", "8", "--gradient-checkpointing",
               "--precision", "fp16", "--device", "cuda", "--warmup-steps", "200",
               "--save-every", "250", "--eval-every", "500", "--log-every", "50", "--seed", "97"]
    log_path = root / f"long_{name}_training.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Long {name} training failed; see {log_path.name}")
    initialization.unlink()
    return output, _final_training_report(log_path.read_text())


def english_samples(torch, checkpoint: Path, root: Path, label: str) -> list[dict]:
    import gc
    from cognition_slm.data import load_jsonl
    from cognition_slm.generate import load_checkpoint, generate_text
    from kaggle_english_run import write_json

    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    rows = []
    for item in load_jsonl(root / "artifacts/broad_eval.jsonl")[:8]:
        rows.append({"prompt": item.prompt, "reference_continuation": item.answer,
                     "generated": generate_text(model, tokenizer, item.prompt,
                                                task_type="language_generation",
                                                max_new_tokens=256, temperature=0, top_k=0)})
    write_json(root / f"{label}_english_samples.json", {"rows": rows, "scope": "Manual fluency review required"})
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Long training must run on Kaggle, never locally")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts")]
    os.environ.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    import torch
    from cognition_slm.data import load_jsonl
    from kaggle_english_run import digest, write_json, compact_completed_stage
    from kaggle_qa_run import evaluate

    started = time.monotonic()
    deadline = started + 10.5 * 3600
    if not torch.cuda.is_available() or (torch.ones(1, device="cuda") + 1).item() != 2:
        raise RuntimeError("Enable a compatible NvidiaTeslaT4 GPU")
    torch.set_num_threads(2)
    if shutil.disk_usage(root).free < 17 * 1024**3:
        raise RuntimeError("Long training needs 17 GiB free working disk")
    manifest = json.loads((root / "source-manifest.json").read_text())
    for filename, expected in manifest.items():
        if digest(root / filename) != expected:
            raise RuntimeError(f"Source hash mismatch: {filename}")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    matches = list(Path("/kaggle/input").rglob("slm-500m-qa-pilot.pt"))
    if len(matches) != 1 or digest(matches[0]) != PARENT_SHA256:
        raise RuntimeError("Attach the completed answer-training run containing the expected pilot weights")
    previous = matches[0]
    (root / "artifacts").mkdir(exist_ok=True)
    report = {"status": "preparing", "parent_sha256": PARENT_SHA256, "source_manifest": manifest,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "stages": []}
    write_json(root / "long_training_report.json", report)
    report["corpus"] = prepare_data(root, previous.parent.parent)
    dev = [(row.to_dict(), [row.answer]) for row in load_jsonl(root / "artifacts/qa_dev.jsonl")]
    report["baseline"] = evaluate(torch, previous, root, "long_baseline_dev", dev)
    report["baseline_english"] = english_samples(torch, previous, root, "baseline")
    for name, seconds in (("english", 5.5 * 3600), ("questions", 3.5 * 3600)):
        budget = min(seconds, deadline - time.monotonic() - 1800)
        if budget < 60:
            report["stop_reason"] = "Session reserve reached before next stage"
            break
        report["status"] = "training_" + name
        write_json(root / "long_training_report.json", report)
        previous, training = train_stage(torch, root, previous, name, budget)
        stage = {"name": name, "training": training, "checkpoint": str(previous)}
        report["stages"].append(stage)
        write_json(root / "long_training_report.json", report)
        stage["development"] = evaluate(torch, previous, root, f"long_{name}_dev", dev)
        stage["english_samples"] = english_samples(torch, previous, root, name)
        if name == "english":
            compact_completed_stage(torch, previous)
        stage["checkpoint_sha256"] = digest(previous)
        write_json(root / "long_training_report.json", report)
    test = [(row.to_dict(), [row.answer]) for row in load_jsonl(root / "artifacts/qa_test.jsonl")]
    report["test"] = evaluate(torch, previous, root, "long_final_test", test)
    from cognition_slm.generate import load_checkpoint, generate_text
    from kaggle_studio_verify import QUESTION_PROBES

    model, tokenizer = load_checkpoint(previous, torch.device("cuda"))
    report["general_questions"] = [
        {"id": name, "prompt": prompt, "expected_rubric": rubric,
         "answer": generate_text(model, tokenizer, prompt, task_type=task,
                                 max_new_tokens=256, temperature=0, top_k=0)}
        for name, prompt, task, rubric in QUESTION_PROBES
    ]
    report.update(status="complete_pending_answer_review", elapsed_seconds=time.monotonic() - started,
                  final_checkpoint=str(previous), final_sha256=digest(previous),
                  metric_scope="Single stored reference per SQuAD question; not directly comparable to earlier multi-reference scores")
    write_json(root / "long_training_report.json", report)
    print("LONG_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
