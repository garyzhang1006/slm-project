"""Score a trained checkpoint on the 24 simple questions, English probes and held-out bits per byte.

It also answers every question in data/everyday_eval.json and scores them under everyday_eval_scores,
per category, as a broader read than the 24-question gate. They do not change the pass gate.

Pass gate: the old 500M elementary checkpoint answered 3 of the 24 holdout questions fully correctly
on manual review (README.md). The new checkpoint passes when more than 3 automatically scored answers
are exact normalized matches. The two "unknown" rows are manual-review only and earn no credit here,
so the gate can only understate the new model. Exact match still needs a human look at the answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.stage2_pretrain import (INPUT, PRETRAIN_EVAL_TOKENS, bits_per_byte, digest,  # noqa: E402
                                     find_one, setup_kaggle, write_json, write_shard)

DEFAULT_CHECKPOINT = "slm-160m-sft.pt"
BASELINE_CORRECT, BASELINE_TOTAL = 3, 24
MAX_NEW_TOKENS = 64
STOP = ("\n",)
# Every SFT record from stage 3 and distill_data carries this tag, so the holdout's two code_explanation
# rows are asked the way the model was trained rather than under a tag it never saw.
TASK_TYPE = "language_generation"
# Everyday requests outside the holdout; rubrics guide the manual read, and looks_english is automatic.
ENGLISH_PROBES = (
    ("probe-01", "Say hello to a new friend in one short sentence.", "a friendly English greeting"),
    ("probe-02", "What color is the sky on a clear day? Reply with one word.", "blue"),
    ("probe-03", "Name a fruit that is yellow. Reply with one word.", "banana or lemon"),
    ("probe-04", "What is 2 plus 2? Reply with the number.", "4"),
    ("probe-05", "Give the opposite of cold. Reply with one word.", "hot or warm"),
    ("probe-06", "Write one sentence about a dog.", "a grammatical sentence about a dog"),
    ("probe-07", "What day comes after Monday?", "Tuesday"),
    ("probe-08", "Give the plural of cat. Reply with one word.", "cats"),
    ("probe-09", "What is your favorite color? I have not told you mine.", "any color; must not claim to know the user's"),
    ("probe-10", "Where do fish live? Answer in one short sentence.", "in water"),
    ("probe-11", "Finish the sentence: The sun rises in the", "east"),
    ("probe-12", "What is my name? I have not told you.", "says it does not know"),
)
_WORD = re.compile(r"[^\W\d_]+|\d+")


def looks_english(text: str) -> bool:
    """Heuristic: mostly printable ASCII and at least half the tokens are ASCII words or numbers."""
    stripped = text.strip()
    if not stripped:
        return False
    printable = sum(32 <= ord(character) < 127 for character in stripped) / len(stripped)
    tokens = _WORD.findall(stripped)
    ascii_words = sum(token.isascii() for token in tokens)
    return printable >= 0.9 and bool(tokens) and ascii_words / len(tokens) >= 0.5


def pass_gate(scores: dict) -> dict:
    total = scores["total"]
    return {"baseline": f"{BASELINE_CORRECT}/{BASELINE_TOTAL} fully correct (old 500M elementary, manual review)",
            "exact": total["exact"], "contains": total["contains"], "scored": total["scored"],
            "manual_review_rows": total["manual_review"], "passed": total["exact"] > BASELINE_CORRECT,
            "rule": f"exact normalized matches > {BASELINE_CORRECT}; unknown rows need manual review"}


def heldout_bits_per_byte(torch, model, tokenizer, device) -> dict:
    from cognition_slm.data import load_pretrain_text, pack_pretrain_text
    from cognition_slm.train import _validation_summary

    evaluations = list(INPUT.rglob("pretrain_eval.jsonl"))
    if len(evaluations) != 1:
        return {"skipped": f"found {len(evaluations)} pretrain_eval.jsonl; attach slm-160m-corpus"}
    subset = ROOT / "artifacts/eval_bpb_subset.jsonl"
    subset.parent.mkdir(exist_ok=True)
    # Same first-2M-token slice the pretrain sessions validate on, so the numbers line up.
    result = write_shard(evaluations[0], subset, 0, PRETRAIN_EVAL_TOKENS)
    encoded = pack_pretrain_text(load_pretrain_text(subset), tokenizer, model.config.block_size)
    summary = _validation_summary(torch, model, encoded, 8, device, 0.0)
    return {**result, "lm_loss": summary["lm_loss"], "packed_rows": summary["records"],
            "bits_per_byte": bits_per_byte(summary["lm_loss"])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT,
                        help="File name to find under /kaggle/input, e.g. slm-160m-pretrain.pt")
    args = parser.parse_args(argv)
    started = time.monotonic()
    torch, manifest = setup_kaggle("Evaluation")
    from cognition_slm.generate import generate_text, load_checkpoint
    from score_holdout import score_predictions

    checkpoint = find_one(args.checkpoint_name)
    device = torch.device("cuda")
    model, tokenizer = load_checkpoint(checkpoint, device)
    model.train(False)
    report_path = ROOT / "eval_report.json"
    report = {"status": "generating", "checkpoint": str(checkpoint), "sha256": digest(checkpoint),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "source_manifest": manifest, "gpu": torch.cuda.get_device_name(0),
              "decoding": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS, "stop": list(STOP),
                           "task_type": TASK_TYPE}}
    holdout = json.loads((ROOT / "data/simple_questions_holdout.json").read_text())["rows"]

    def answer(prompt: str) -> str:
        return generate_text(model, tokenizer, prompt, task_type=TASK_TYPE, max_new_tokens=MAX_NEW_TOKENS,
                             temperature=0, top_k=0, stop_sequences=list(STOP))

    report["simple_questions"] = [{**row, "answer": answer(row["prompt"])} for row in holdout]
    report["simple_questions_scores"] = score_predictions(holdout, report["simple_questions"])
    report["pass_gate"] = pass_gate(report["simple_questions_scores"])
    everyday = json.loads((ROOT / "data/everyday_eval.json").read_text())["rows"]
    report["everyday_eval"] = [{**row, "answer": answer(row["prompt"])} for row in everyday]
    report["everyday_eval_scores"] = score_predictions(everyday, report["everyday_eval"])
    report["english_probes"] = []
    for identifier, prompt, rubric in ENGLISH_PROBES:
        text = answer(prompt)
        report["english_probes"].append({"id": identifier, "prompt": prompt, "expected_rubric": rubric,
                                         "answer": text, "looks_english": looks_english(text)})
    answers = [row["answer"] for row in report["simple_questions"] + report["english_probes"]]
    report["looks_english_rate"] = sum(map(looks_english, answers)) / len(answers)
    report["status"] = "bits_per_byte"
    write_json(report_path, report)
    report["heldout_text"] = heldout_bits_per_byte(torch, model, tokenizer, device)
    report.update(status="complete_pending_manual_review", elapsed_seconds=round(time.monotonic() - started, 1))
    write_json(report_path, report)
    print("EVAL_COMPLETE", json.dumps({"pass_gate": report["pass_gate"],
                                       "looks_english_rate": report["looks_english_rate"],
                                       "everyday_exact": report["everyday_eval_scores"]["total"]["exact_accuracy"],
                                       "bits_per_byte": report["heldout_text"].get("bits_per_byte")}), flush=True)


if __name__ == "__main__":
    main()
