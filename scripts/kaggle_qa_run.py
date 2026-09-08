"""Kaggle-only answer training, with a measured pilot before the longer run."""

from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
PARENT_SHA256 = "f1174d8f87b7f6cce734dd15ae0b5b722b964bf3be2f6c8dae923f0dd7cdbaa3"


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def context_key(raw: dict) -> str:
    return hashlib.sha256(normalized(raw["context"]).encode()).hexdigest()


def prepare_data(root: Path, parent: Path) -> tuple[list, list, dict]:
    from datasets import load_dataset
    from cognition_slm.data import load_jsonl
    from build_curriculum_data import write_jsonl
    from english_corpus import deterministic_split
    from qa_corpus import squad_record
    from kaggle_studio_verify import QUESTION_PROBES
    from kaggle_english_run import digest, write_json

    validation = []
    for raw in load_dataset("rajpurkar/squad", revision=SQUAD_REVISION,
                            split="validation", streaming=True):
        row = squad_record(raw)
        if row:
            validation.append((context_key(raw), raw, row))
    validation.sort(key=lambda item: (item[0], item[1]["id"]))
    # Contexts, not individual questions, define the development/test partition.
    dev = [(row, raw["answers"]["text"]) for key, raw, row in validation if int(key, 16) % 2 == 0][:64]
    test = [(row, raw["answers"]["text"]) for key, raw, row in validation if int(key, 16) % 2 == 1][:128]
    if len(dev) != 64 or len(test) != 128:
        raise RuntimeError("Insufficient complete SQuAD holdouts within the byte budget")
    excluded_contexts = {key for key, _, _ in validation}
    reserved = {normalized(raw["question"]) for _, raw, _ in validation}
    reserved.update(normalized(prompt) for _, prompt, _, _ in QUESTION_PROBES)
    dolly = [row.to_dict() for row in load_jsonl(parent / "artifacts/questions_train.jsonl")
             if row.source == "databricks/databricks-dolly-15k"
             and len(row.answer.encode()) <= 192 and normalized(row.prompt) not in reserved]
    if len(dolly) < 1000:
        raise RuntimeError("Expected at least 1000 short Dolly answers in the parent training split")
    squad = []
    for raw in load_dataset("rajpurkar/squad", revision=SQUAD_REVISION, split="train", streaming=True):
        if context_key(raw) in excluded_contexts or normalized(raw["question"]) in reserved:
            continue
        row = squad_record(raw)
        if row:
            squad.append(row)
    squad, _ = deterministic_split(squad, eval_count=0)
    squad = squad[:min(20_000, 2 * len(dolly))]
    if len(squad) < 2000:
        raise RuntimeError("Too few SQuAD training examples survived validation")
    train, _ = deterministic_split(squad + dolly, eval_count=0)
    write_jsonl(train, root / "artifacts/qa_train.jsonl")
    write_jsonl([row for row, _ in dev], root / "artifacts/qa_dev.jsonl")
    write_jsonl([row for row, _ in test], root / "artifacts/qa_test.jsonl")
    for split in ("dev", "test"):
        subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train",
                        "artifacts/qa_train.jsonl", "--eval", f"artifacts/qa_{split}.jsonl"], check=True)
    report = {"squad_revision": SQUAD_REVISION, "squad_train": len(squad), "dolly_train": len(dolly),
              "train": len(train), "dev": len(dev), "test": len(test),
              "scope": "Complete short passages; official SQuAD train/validation with context overlap excluded and context-grouped dev/test",
              "hashes": {split: digest(root / f"artifacts/qa_{split}.jsonl") for split in ("train", "dev", "test")}}
    write_json(root / "qa_corpus_manifest.json", report)
    return dev, test, report


def evaluate(torch, checkpoint: Path, root: Path, label: str, items: list) -> dict:
    from cognition_slm.generate import load_checkpoint, generate_text
    from cognition_slm.data import validate_record, encode_examples
    from cognition_slm.train import _validation_summary, _batch
    from kaggle_english_run import write_json
    from qa_corpus import answer_scores

    device = torch.device("cuda")
    model, tokenizer = load_checkpoint(checkpoint, device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 499_524_075:
        raise RuntimeError(f"Expected 499524075 parameters, got {parameters}")
    model.eval()
    examples = [validate_record(row) for row, _ in items[:16]]
    shuffled = [replace(row, prompt=examples[(i + 1) % len(examples)].prompt) for i, row in enumerate(examples)]
    result = {"parameters": parameters, "conditioning": {}, "rows": []}
    for name, rows in (("correct_prompt", examples), ("shuffled_prompt", shuffled)):
        encoded = encode_examples(rows, tokenizer, model.config.block_size)
        result["conditioning"][name] = _validation_summary(torch, model, encoded, 1, device)["lm_loss"]
        if name == "correct_prompt":
            correct = 0
            with torch.no_grad():
                for index, item in enumerate(encoded):
                    batch = _batch(torch, encoded, [index], device)
                    logits = model(batch[0], attention_mask=batch[1]).logits
                    start = item["answer_start"]
                    correct += int(logits[0, start - 1].argmax().item() == item["input_ids"][start])
            result["first_answer_byte_accuracy"] = correct / len(encoded)
    for index, (row, references) in enumerate(items):
        answer = generate_text(model, tokenizer, row["prompt"], task_type="language_generation",
                               max_new_tokens=128, temperature=0, top_k=0)
        scored = {"id": row["id"], "prompt": row["prompt"], "references": references,
                  "answer": answer, **answer_scores(answer, references)}
        if index < 8:
            torch.manual_seed(71 + index)
            sampled = generate_text(model, tokenizer, row["prompt"], task_type="language_generation",
                                    max_new_tokens=128, temperature=0.7, top_k=40)
            scored["sampled"] = {"answer": sampled, **answer_scores(sampled, references)}
        result["rows"].append(scored)
        write_json(root / f"{label}.json", result)
    for metric in ("exact_match", "token_f1"):
        result[metric] = sum(row[metric] for row in result["rows"]) / len(items)
    write_json(root / f"{label}.json", result)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_stage(torch, root: Path, source: Path, name: str, steps: int) -> Path:
    from kaggle_english_run import initialize_stage, compact_completed_stage

    initialization = root / "artifacts/qa_initialization.pt"
    initialize_stage(torch, source, initialization)
    output = root / f"artifacts/slm-500m-qa-{name}.pt"
    command = [sys.executable, "-m", "cognition_slm.train", "--resume", str(initialization),
               "--data", "artifacts/qa_train.jsonl", "--eval-data", "artifacts/qa_dev.jsonl",
               "--out", str(output), "--steps", str(steps), "--learning-rate", "0.00005",
               "--batch-size", "1", "--gradient-accumulation-steps", "8", "--gradient-checkpointing",
               "--precision", "fp16", "--device", "cuda", "--warmup-steps", "50",
               "--save-every", "250", "--eval-every", "250", "--log-every", "50", "--seed", "71"]
    with (root / f"qa_{name}_training.log").open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"QA {name} failed; inspect qa_{name}_training.log")
    initialization.unlink()
    compact_completed_stage(torch, output)
    return output


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("QA training and model evaluation must run on Kaggle, never locally")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts")]
    os.environ.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    import torch
    from kaggle_english_run import digest, write_json
    from qa_corpus import pilot_improved

    if not torch.cuda.is_available():
        raise RuntimeError("Enable NvidiaTeslaT4 on Kaggle")
    if (torch.ones(1, device="cuda") + 1).item() != 2:
        raise RuntimeError("GPU arithmetic check failed")
    torch.set_num_threads(2)
    if shutil.disk_usage(root).free < 16 * 1024**3:
        raise RuntimeError("QA training needs 16 GiB free in /kaggle/working")
    manifest = json.loads((root / "source-manifest.json").read_text())
    for filename, expected in manifest.items():
        if digest(root / filename) != expected:
            raise RuntimeError(f"Source hash mismatch: {filename}")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    matches = list(Path("/kaggle/input").rglob("slm-500m-questions.pt"))
    if len(matches) != 1 or digest(matches[0]) != PARENT_SHA256:
        raise RuntimeError("Attach the completed English corpus v2 output with the expected question weights")
    checkpoint = matches[0]
    parent = checkpoint.parent.parent
    (root / "artifacts").mkdir(exist_ok=True)
    dev, test, corpus = prepare_data(root, parent)
    report = {"status": "baseline", "source_manifest": manifest, "parent_sha256": PARENT_SHA256,
              "corpus": corpus, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    write_json(root / "qa_training_report.json", report)
    report["baseline"] = evaluate(torch, checkpoint, root, "qa_baseline_dev", dev)
    report["status"] = "pilot_training"
    write_json(root / "qa_training_report.json", report)
    pilot = train_stage(torch, root, checkpoint, "pilot", 500)
    report["pilot"] = evaluate(torch, pilot, root, "qa_pilot_dev", dev)
    report["pilot_sha256"] = digest(pilot)
    report["pilot_gate_passed"] = pilot_improved(report["baseline"], report["pilot"])
    if report["pilot_gate_passed"]:
        report["status"] = "extended_training"
        write_json(root / "qa_training_report.json", report)
        candidate = train_stage(torch, root, pilot, "extended", 4000)
        report["extended"] = evaluate(torch, candidate, root, "qa_extended_dev", dev)
        # Development selection never sees the final test answers.
        if (report["extended"]["exact_match"], report["extended"]["token_f1"]) < (
                report["pilot"]["exact_match"], report["pilot"]["token_f1"]):
            candidate = pilot
    else:
        candidate = checkpoint
        report["stop_reason"] = "Pilot did not improve held-out answers enough; extended training skipped"
    report["test"] = evaluate(torch, candidate, root, "qa_final_test", test)
    from cognition_slm.generate import load_checkpoint, generate_text
    from kaggle_studio_verify import QUESTION_PROBES

    model, tokenizer = load_checkpoint(candidate, torch.device("cuda"))
    report["original_question_probes"] = [
        {"id": name, "prompt": prompt, "expected_rubric": rubric,
         "answer": generate_text(model, tokenizer, prompt, task_type=task,
                                 max_new_tokens=256, temperature=0, top_k=0)}
        for name, prompt, task, rubric in QUESTION_PROBES
    ]
    report.update(status="complete_pending_answer_review" if report["pilot_gate_passed"] else "stopped_no_answer_gain",
                  candidate=str(candidate),
                  candidate_sha256=digest(candidate))
    write_json(root / "qa_training_report.json", report)
    print("QA_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
