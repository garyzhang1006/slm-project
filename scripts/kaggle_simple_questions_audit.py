"""Record unseen-wording simple-question answers on Kaggle, without training."""

import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this audit on Kaggle; never execute the model locally")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ["OMP_NUM_THREADS"] = "2"
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Source hash mismatch: {name}")
    matches = list(Path("/kaggle/input").rglob("slm-500m-efficient.pt"))
    if len(matches) != 1:
        raise RuntimeError("Attach the completed efficient-continuation kernel")
    checkpoint = matches[0]
    parent = json.loads((checkpoint.parent.parent / "efficient_report.json").read_text())
    if parent.get("status") != "complete_pending_answer_review" or digest(checkpoint) != parent.get("sha256"):
        raise RuntimeError("Completed checkpoint and report must match before evaluation")
    holdout_path = root / "data/simple_questions_holdout.json"
    holdout = json.loads(holdout_path.read_text())
    training_prompts = set()
    normalize = lambda text: " ".join(text.casefold().split())
    for line in (checkpoint.parent / "broad_train.jsonl").read_text().splitlines():
        if line.strip():
            training_prompts.add(normalize(json.loads(line)["prompt"]))
    for row in holdout["rows"]:
        if normalize(row["prompt"]) in training_prompts:
            raise RuntimeError(f"Evaluation prompt duplicates training: {row['id']}")
    import torch
    from cognition_slm.generate import load_checkpoint, generate_text
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    torch.set_num_threads(2)
    model, tokenizer = load_checkpoint(checkpoint, torch.device("cuda"))
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 499_524_075:
        raise RuntimeError(f"Unexpected model parameter count: {parameters}")
    report = {"checkpoint_sha256": parent["sha256"], "holdout_sha256": digest(holdout_path),
              "parameters": parameters, "status": "evaluating", "rows": [],
              "scope": "Manual rubric review required; no training or generated-code execution"}
    destination = root / "simple_questions_audit.json"
    for row in holdout["rows"]:
        answer = generate_text(model, tokenizer, row["prompt"], task_type=row["task_type"],
                               temperature=0, top_k=0, max_new_tokens=160)
        report["rows"].append({**row, "answer": answer})
        destination.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete_pending_manual_review"
    destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
