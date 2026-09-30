"""Pull every stage's JSON report from Kaggle and write compute/RESULTS.md.

Only the small report files are downloaded (never checkpoints), and nothing runs a model. LoRA eval
predictions are re-scored here against the current answer keys in data/, so fixes to accepted_answers
show up without another GPU run. Run from the repo root, logged in with the Kaggle CLI, e.g.
    python3 compute/collect_results.py --owner YOUR_KAGGLE_USERNAME
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.run_pipeline import MAX_PRETRAIN_SESSIONS, Kaggle  # noqa: E402
from compute.stages import stage_slug  # noqa: E402
from scripts.score_holdout import REPORT_KEYS, score_predictions  # noqa: E402

EVAL_FILES = ("data/everyday_eval.json", "data/simple_questions_holdout.json")
OUTPUT = ROOT / "compute" / "RESULTS.md"


def collect(fetch) -> dict:
    """fetch(slug, filename) -> dict | None. Pretrain sessions are read until the first one without a report."""
    pretrain = []
    for session in range(1, MAX_PRETRAIN_SESSIONS + 1):
        report = fetch(stage_slug("pretrain", session), f"pretrain_session_{session}.json")
        if not report:
            break
        pretrain.append(report)
    return {"pretrain": pretrain,
            "lora": fetch(stage_slug("lora"), "lora_report.json"),
            "lora_eval": fetch(stage_slug("lora_eval"), "lora_eval_report.json"),
            "lora_1b7": fetch(stage_slug("lora_1b7"), "lora_report.json"),
            "lora_1b7_eval": fetch(stage_slug("lora_1b7_eval"), "lora_eval_report.json"),
            "distill": fetch(stage_slug("distill_data"), "distill_manifest.json"),
            "sft": fetch(stage_slug("sft"), "sft_report.json"),
            "eval": fetch(stage_slug("eval"), "eval_report.json")}


def rescore(lora_eval: dict, root: Path = ROOT) -> dict:
    """{set key: {"base": scores, "lora": scores}} against today's answer keys."""
    scored = {}
    for name in EVAL_FILES:
        key = REPORT_KEYS[Path(name).name]
        rows = json.loads((root / name).read_text())["rows"]
        sides = {side: lora_eval.get(side, {}).get(key, {}).get("predictions") for side in ("base", "lora")}
        if all(sides.values()):
            scored[key] = {side: score_predictions(rows, predictions) for side, predictions in sides.items()}
    return scored


def number(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def exact(scores: dict) -> str:
    total = scores["total"]
    return f"{total['exact']}/{total['scored']}"


def contains(scores: dict) -> str:
    total = scores["total"]
    return f"{total['contains']}/{total['scored']}"


def lora_sections(name: str, lora: dict | None, lora_eval: dict | None, rescored: dict) -> list[str]:
    """Adapter training summary plus its re-scored eval against the base model."""
    lines = ["", f"## LoRA adapter ({name})", ""]
    if lora:
        training = lora.get("training", {})
        baseline, final = lora.get("baseline", {}).get("scores"), lora.get("final", {}).get("scores")
        lines += [f"- Status: {lora.get('status')}",
                  f"- Holdout exact: base {exact(baseline) if baseline else 'n/a'}, "
                  f"adapter {exact(final) if final else 'n/a'}",
                  f"- Eval loss: {number(training.get('initial_eval_loss'))} before training, best "
                  f"{number(training.get('best_eval_loss'))} at step {training.get('best_step', 'n/a')}, final "
                  f"{number(training.get('final_eval_loss'))}",
                  f"- Adapter sha256: `{lora.get('adapter_sha256') or 'not recorded (older runner)'}`"]
    else:
        lines.append("No LoRA report yet.")

    lines += ["", f"## {name}: LoRA vs base, re-scored with the current answer keys", ""]
    everyday = rescored.get("everyday_eval")
    if everyday:
        used, current = lora_eval.get("adapter_sha256"), (lora or {}).get("adapter_sha256")
        if used and current and used != current:
            lines += ["These predictions came from a different adapter than the LoRA report above.", ""]
        for key, scores in rescored.items():
            lines.append(f"- {key}: base {exact(scores['base'])} exact and {contains(scores['base'])} contains, "
                         f"LoRA {exact(scores['lora'])} exact and {contains(scores['lora'])} contains")
        lines += ["", "Exact means the whole reply is an accepted answer. Contains means an accepted answer appears "
                  "as whole words in the reply, which credits full-sentence answers such as \"Water freezes at 0 "
                  "degrees Celsius.\" but can also credit a reply that names the answer and then contradicts it.",
                  "", "| category | base exact | LoRA exact | base contains | LoRA contains | scored |",
                  "|---|---|---|---|---|---|"]
        base, adapter = everyday["base"]["categories"], everyday["lora"]["categories"]
        for name in sorted(set(base) | set(adapter)):
            empty = {"exact": 0, "contains": 0, "scored": 0}
            left, right = base.get(name, empty), adapter.get(name, empty)
            lines.append(f"| {name} | {left['exact']} | {right['exact']} | {left['contains']} | "
                         f"{right['contains']} | {right['scored']} |")
        manual = [row for scores in rescored.values() for row in scores["lora"].get("rows", [])
                  if row.get("manual_review")]
        predictions = {row["id"]: row for side in lora_eval.get("lora", {}).values()
                       for row in side.get("predictions", [])}
        if manual:
            lines += ["", "These rows ask the model to admit it doesn't know, so a person judges them and they "
                      "are left out of the counts:", "", "| id | prompt | LoRA answer |", "|---|---|---|"]
            for row in manual:
                shown = predictions.get(row["id"], {})
                cells = [row["id"], shown.get("prompt", ""), shown.get("answer", "")]
                # split() folds every line break, a lone "\r" included, so a model answer stays on one row.
                lines.append("| " + " | ".join(" ".join(cell.replace("|", "/").split()) for cell in cells) + " |")
    else:
        lines.append("No finished lora_eval report yet.")

    return lines


def render(results: dict, rescored: dict, stamp: str, large_rescored: dict | None = None) -> str:
    lines = ["# Results", "", f"Collected from Kaggle kernel reports on {stamp} by `compute/collect_results.py`. "
             "Regenerate it rather than editing by hand.", ""]

    lines += ["## slm-160m pretraining", ""]
    if results["pretrain"]:
        lines += ["| session | steps reached | s/step | eval bits/byte | status |", "|---|---|---|---|---|"]
        lines += [f"| {report.get('session')} | {report.get('step_reached')} | "
                  f"{number(report.get('seconds_per_step'), 2)} | {number(report.get('eval_bits_per_byte'))} | "
                  f"{report.get('status')} |" for report in results["pretrain"]]
    else:
        lines.append("No finished pretrain session yet.")

    lines += lora_sections("SmolLM2-360M-Instruct", results["lora"], results["lora_eval"], rescored)
    lines += lora_sections("SmolLM2-1.7B-Instruct", results.get("lora_1b7"), results.get("lora_1b7_eval"),
                           large_rescored or {})

    lines += ["", "## Distilled answers", ""]
    distill = results["distill"]
    if distill:
        dropped = ", ".join(f"{reason} {count}" for reason, count in sorted(distill.get("dropped", {}).items()))
        lines.append(f"{distill.get('rows')} rows kept from {distill.get('prompts')} prompts"
                     + (f" (dropped: {dropped})." if dropped else "."))
    else:
        lines.append("No distill_data manifest yet.")

    lines += ["", "## slm-160m SFT", ""]
    sft = results["sft"]
    if sft:
        steps = sft.get("training", {}).get("steps")
        lines.append(f"Status {sft.get('status')}, built on pretrain session {sft.get('pretrain_session')}, "
                     f"{steps} steps, eval LM loss {number(sft.get('eval_lm_loss'))}.")
    else:
        lines.append("No SFT report yet.")

    lines += ["", "## slm-160m eval", ""]
    evaluation = results["eval"]
    if evaluation:
        lines += [f"- Holdout exact: {exact(evaluation['simple_questions_scores'])}"
                  if evaluation.get("simple_questions_scores") else "- Holdout exact: n/a",
                  f"- Everyday exact: {exact(evaluation['everyday_eval_scores'])}"
                  if evaluation.get("everyday_eval_scores") else "- Everyday exact: n/a",
                  f"- Looks-English rate: {number(evaluation.get('looks_english_rate'))}",
                  f"- Held-out text bits/byte: {number(evaluation.get('heldout_text', {}).get('bits_per_byte'))}"]
    else:
        lines.append("No eval report yet.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle username that owns the kernels")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    results = collect(Kaggle(args.owner).report)
    rescored = rescore(results["lora_eval"]) if results["lora_eval"] else {}
    large = rescore(results["lora_1b7_eval"]) if results["lora_1b7_eval"] else {}
    args.out.write_text(render(results, rescored, time.strftime("%Y-%m-%d %H:%M"), large))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
