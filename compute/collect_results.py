"""Pull every stage's JSON report from Kaggle and write compute/RESULTS.md.

Only the small report files are downloaded (never checkpoints), and nothing runs a model. LoRA eval
predictions are re-scored here against the current answer keys in data/, so fixes to accepted_answers
show up without another GPU run. Run from the repo root, logged in with the Kaggle CLI, e.g.
    python3 compute/collect_results.py --owner YOUR_KAGGLE_USERNAME
or read the reports from kernel outputs already on disk, as the results kernel does with
    python3 compute/collect_results.py --input-dir /kaggle/input
"""

from __future__ import annotations

import argparse
import hashlib
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
REPORTS = ROOT / "reports"
# Per-question answers stay in the repo, so an answer-key audit can read them without a Kaggle download.
PREDICTION_FILES = {"lora_eval": "lora_eval_360m.json", "lora_1b7_eval": "lora_eval_1b7.json"}


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


def local_report(input_dir: Path, slug: str, filename: str) -> dict | None:
    """The one filename under a folder named slug in input_dir, read like Kaggle.report reads a download."""
    # Kaggle mounts each attached output at /kaggle/input/notebooks/<owner>/<slug>/, and slm-lora-baseline and
    # slm-lora-1b7 both hold lora_report.json, so the file name alone would find two.
    matches = [path for path in input_dir.rglob(filename)
               if slug in path.relative_to(input_dir).parts[:-1]] if input_dir.is_dir() else []
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0].read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


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


def predictions_record(stage: str, lora_eval: dict) -> dict:
    """The kernel, adapter and every {id, answer} per eval file and side, in the report's order."""
    answers = {}
    for name in EVAL_FILES:
        key = REPORT_KEYS[Path(name).name]
        found = {side: lora_eval.get(side, {}).get(key, {}).get("predictions") for side in ("base", "lora")}
        sides = {side: [{"id": row["id"], "answer": row["answer"]} for row in rows]
                 for side, rows in found.items() if rows}
        if sides:
            answers[name] = sides
    return {"kernel": stage_slug(stage), "adapter_sha256": lora_eval.get("adapter_sha256"), "predictions": answers}


def finished(lora_eval: dict | None) -> bool:
    """lora_eval.py rewrites its report after every set, so a run that died partway leaves some sets unanswered."""
    return (lora_eval or {}).get("status") == "complete_pending_manual_review"


def write_predictions(results: dict, directory: Path) -> list[Path]:
    """Writes a file only for a kernel whose finished report came back, so a failed download or a run that
    died partway keeps the last answers."""
    written = []
    for stage, filename in PREDICTION_FILES.items():
        if not finished(results.get(stage)):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        # Sorted keys and one field per line keep a re-collection's git diff down to the answers that changed.
        path.write_text(json.dumps(predictions_record(stage, results[stage]), indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


def key_hashes(root: Path = ROOT) -> dict:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in EVAL_FILES}


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
        lines += ["", "Exact means the whole reply is an accepted answer, or sentences that each are one. An abstain "
                  "reply is also exact when each clause only refuses, naming at most what it does not know, as in "
                  "\"I don't know your name. You haven't told me.\" Contains means an accepted answer appears "
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


def render(results: dict, rescored: dict, stamp: str, large_rescored: dict | None = None,
           root: Path = ROOT) -> str:
    keys = ", ".join(f"`{name}` {digest[:12]}" for name, digest in key_hashes(root).items())
    lines = ["# Results", "", f"Collected from Kaggle kernel reports on {stamp} by `compute/collect_results.py`. "
             "Regenerate it rather than editing by hand.", "",
             f"Re-scored with these answer keys (first 12 hex characters of each file's sha256): {keys}.", ""]

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
        # eval_lm_loss belongs to the shipped best checkpoint, which can sit well before the last step trained.
        best = f" from step {sft['best_step']}" if sft.get("best_step") is not None else ""
        lines.append(f"Status {sft.get('status')}, built on pretrain session {sft.get('pretrain_session')}, "
                     f"{steps} steps, eval LM loss {number(sft.get('eval_lm_loss'))}{best}.")
    else:
        lines.append("No SFT report yet.")

    lines += ["", "## slm-160m eval", ""]
    evaluation = results["eval"]
    if evaluation:
        # A re-pushed sft replaces the checkpoint before eval reruns on it.
        used, current = evaluation.get("sha256"), (sft or {}).get("sha256")
        if used and current and used != current:
            lines += ["These scores came from an older SFT checkpoint than the SFT report above.", ""]
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--owner", help="Kaggle username that owns the kernels")
    source.add_argument("--input-dir", type=Path,
                        help="read the reports from kernel outputs under this folder instead of the Kaggle CLI")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS,
                        help="where the per-question LoRA eval answers are written")
    args = parser.parse_args(argv)
    if args.input_dir is None:
        results = collect(Kaggle(args.owner).report)
    elif args.input_dir.is_dir():
        results = collect(lambda slug, filename: local_report(args.input_dir, slug, filename))
    else:
        # Every report would read as missing, and RESULTS.md would lose every number it has.
        parser.error(f"--input-dir {args.input_dir} is not a folder")
    for path in write_predictions(results, args.reports_dir):
        print(f"wrote {path}")
    # The same test as write_predictions, so RESULTS.md never scores answers that reports/ did not save.
    rescored = rescore(results["lora_eval"]) if finished(results["lora_eval"]) else {}
    large = rescore(results["lora_1b7_eval"]) if finished(results["lora_1b7_eval"]) else {}
    args.out.write_text(render(results, rescored, time.strftime("%Y-%m-%d %H:%M"), large))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
