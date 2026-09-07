"""Continue training the custom 499M model on English text, exclusively on Kaggle."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SOURCES = {
    "roneneldan/TinyStories": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    "databricks/databricks-dolly-15k": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
}
PARENT_SHA256 = "a2f5aca4607c9d35be7a35063045ca8d7f1bc90108e7906b072646d7401dce77"


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


def prepare_data(root: Path) -> dict:
    from datasets import load_dataset
    from build_curriculum_data import write_jsonl
    from english_corpus import deterministic_split, dolly_record, story_record
    from kaggle_studio_verify import QUESTION_PROBES

    reserved = {" ".join(prompt.casefold().split()) for _, prompt, _, _ in QUESTION_PROBES}
    report = {"sources": SOURCES, "splits": {}}
    splits = {}
    for name, source in (("english", "roneneldan/TinyStories"),
                         ("questions", "databricks/databricks-dolly-15k")):
        rows = []
        scanned = 0
        stream = load_dataset(source, revision=SOURCES[source], split="train", streaming=True)
        for index, raw in enumerate(stream):
            scanned += 1
            row = (story_record(raw.get("text"), str(index), max_bytes=768)
                   if name == "english" else dolly_record(raw, index, max_bytes=1024))
            if row and " ".join(row["prompt"].casefold().split()) not in reserved:
                rows.append(row)
            if name == "english" and (len(rows) >= 30_000 or scanned >= 400_000):
                break
        train, evaluation = deterministic_split(rows, eval_count=128)
        minimum = 10_000 if name == "english" else 1_000
        if len(train) < minimum:
            raise RuntimeError(f"{source}: only {len(train)} usable training rows; need {minimum}")
        splits[name] = (train, evaluation)
        report["splits"][name] = {"scanned": scanned, "accepted": len(rows),
                                  "train": len(train), "eval": len(evaluation)}

    # Retain some fluency practice during instruction tuning; never replay holdouts.
    question_train, question_eval = splits["questions"]
    question_train.extend(splits["english"][0][:len(question_train) // 4])
    all_eval = splits["english"][1] + question_eval
    held_out = {" ".join(row["prompt"].casefold().split()) for row in all_eval}
    for name, (train, evaluation) in splits.items():
        if any(" ".join(row["prompt"].casefold().split()) in held_out for row in train):
            raise RuntimeError("Cross-stage training/evaluation prompt overlap")
        for split, rows in (("train", train), ("eval", evaluation)):
            path = root / "artifacts" / f"{name}_{split}.jsonl"
            write_jsonl(rows, path)
            report["splits"][name][split + "_sha256"] = digest(path)
            report["splits"][name][split + "_with_replay"] = len(rows)
        subprocess.run([sys.executable, "-m", "cognition_slm.audit", "--train",
                        f"artifacts/{name}_train.jsonl", "--eval", f"artifacts/{name}_eval.jsonl"], check=True)
    write_json(root / "corpus_manifest.json", report)
    return report


def initialize_stage(torch, source: Path, destination: Path) -> None:
    from cognition_slm.checkpoint import load_checkpoint_payload

    payload, config = load_checkpoint_payload(torch, source, inference_only=True)
    count = sum(tensor.numel() for tensor in payload["model_state_dict"].values())
    # Tied embeddings can appear twice in a state dict; verify actual count in probe().
    if config.block_size != 2048 or count < 499_000_000:
        raise RuntimeError("Attached weights do not match the custom 500M/2048 configuration")
    torch.save({"model_config": payload["model_config"],
                "model_state_dict": payload["model_state_dict"],
                "metadata": {"initialization_source": str(source),
                             "parent_metadata": payload.get("metadata", {}),
                             "optimizer_reset_reason": "new corpus and learning-rate schedule"}}, destination)
    del payload
    gc.collect()


def compact_completed_stage(torch, checkpoint: Path) -> None:
    from cognition_slm.checkpoint import load_checkpoint_payload

    payload, _ = load_checkpoint_payload(torch, checkpoint, inference_only=True)
    temporary = checkpoint.with_suffix(".compact.pt")
    torch.save(payload, temporary)
    del payload
    gc.collect()
    temporary.replace(checkpoint)


def probe(torch, checkpoint: Path, root: Path, label: str) -> dict:
    from cognition_slm.generate import generate_text, load_checkpoint
    from cognition_slm.data import encode_examples, load_jsonl
    from cognition_slm.train import _validation_summary
    from kaggle_studio_verify import QUESTION_PROBES

    device = torch.device("cuda")
    model, tokenizer = load_checkpoint(checkpoint, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 499_524_075:
        raise RuntimeError(f"Expected 499524075 parameters, got {parameters}")
    result = {"parameters": parameters, "context_window": model.config.block_size,
              "checkpoint_sha256": digest(checkpoint), "validation": {}, "answers": [],
              "scope": "Exact prompt wording excluded from new corpus; manual answer review required"}
    for name in ("english", "questions"):
        encoded = encode_examples(load_jsonl(root / f"artifacts/{name}_eval.jsonl"),
                                  tokenizer, model.config.block_size)
        result["validation"][name] = _validation_summary(torch, model, encoded, 1, device)
    for name, prompt, task, rubric in QUESTION_PROBES:
        text = generate_text(model, tokenizer, prompt, task_type=task,
                             max_new_tokens=256, temperature=0, top_k=0)
        result["answers"].append({"id": name, "prompt": prompt, "answer": text,
                                   "expected_rubric": rubric})
        write_json(root / f"{label}_evaluation.json", result)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Training and evaluation must run on Kaggle, never locally")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    os.environ.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    import torch
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets in the Kaggle environment, then rerun") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before starting English training")
    torch.set_num_threads(2)
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    checkpoints = list(Path("/kaggle/input").rglob("slm-500m-language-quality.pt"))
    if len(checkpoints) != 1 or digest(checkpoints[0]) != PARENT_SHA256:
        raise RuntimeError("Attach the original slm-500m-english-code-quality-v2 kernel output")
    (root / "artifacts").mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 16 * 1024**3:
        raise RuntimeError("English training needs at least 16 GiB free in /kaggle/working")
    started = time.monotonic()
    report = {"status": "preparing", "source_manifest": manifest, "parent_sha256": PARENT_SHA256,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "datasets": datasets.__version__, "stages": []}
    report["corpus"] = prepare_data(root)
    report["baseline"] = probe(torch, checkpoints[0], root, "baseline")
    previous = checkpoints[0]
    initialization = root / "artifacts/stage_initialization.pt"
    for name, steps, learning_rate in (("english", 4000, "0.0002"), ("questions", 1000, "0.00005")):
        report["status"] = "training_" + name
        write_json(root / "english_training_report.json", report)
        initialize_stage(torch, previous, initialization)
        output = root / f"artifacts/slm-500m-{name}.pt"
        command = [sys.executable, "-m", "cognition_slm.train", "--resume", str(initialization),
                   "--data", f"artifacts/{name}_train.jsonl", "--eval-data", f"artifacts/{name}_eval.jsonl",
                   "--out", str(output), "--steps", str(steps), "--learning-rate", learning_rate,
                   "--batch-size", "1", "--gradient-accumulation-steps", "8", "--gradient-checkpointing",
                   "--precision", "fp16", "--device", "cuda", "--warmup-steps", "100",
                   "--save-every", "250", "--eval-every", "500", "--log-every", "50", "--seed", "37"]
        with (root / f"{name}_training.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"{name} training failed; see {name}_training.log and saved checkpoint")
        previous = output
        initialization.unlink()
        report["stages"].append({"name": name, "steps": steps, "checkpoint": str(output),
                                 "evaluation": probe(torch, output, root, name)})
        compact_completed_stage(torch, output)
        report["stages"][-1].update(checkpoint_format="model_only_completed_stage",
                                      checkpoint_sha256=digest(output))
        write_json(root / "english_training_report.json", report)
    report.update(status="complete_pending_answer_review", elapsed_seconds=round(time.monotonic() - started, 2))
    write_json(root / "english_training_report.json", report)
    print("ENGLISH_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
