"""Automatic exact and contains-match scoring for data/simple_questions_holdout.json or any file in its schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDOUT = ROOT / "data" / "simple_questions_holdout.json"
# compute/stage5_evaluate.py stores each file's answers under its own report key.
REPORT_KEYS = {"simple_questions_holdout.json": "simple_questions", "everyday_eval.json": "everyday_eval"}
# These rubrics describe a behavior (abstaining) rather than an answer string, so a human judges them.
MANUAL_CATEGORIES = frozenset({"unknown"})
_UNITS = {word: index for index, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {word: 10 * index for index, word in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_TOKENS = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+")


def normalize_answer(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace and spell numbers below 100 as digits."""
    tokens = _TOKENS.findall(text.casefold().replace("'", "").replace("’", ""))
    result = []
    for token in tokens:
        # "twenty one" (or "twenty-one", split on the hyphen) folds into the preceding tens value.
        if token in _UNITS and 0 < _UNITS[token] < 10 and result and result[-1][1] in _TENS:
            result[-1] = (str(_TENS[result[-1][1]] + _UNITS[token]), None)
        elif token in _UNITS or token in _TENS:
            result.append((str(_UNITS.get(token, _TENS.get(token))), token))
        else:
            result.append((token, None))
    return " ".join(value for value, _ in result)


def accepted_answers(row: dict) -> list[str]:
    """Rows may list accepted_answers; the current holdout only has expected_rubric."""
    answers = row.get("accepted_answers") or [row["expected_rubric"]]
    return [normalize_answer(answer) for answer in answers]


def score_answer(row: dict, answer: str) -> dict | None:
    """Return exact/contains flags, or None when the row needs manual review."""
    if row["category"] in MANUAL_CATEGORIES:
        return None
    predicted = normalize_answer(answer)
    padded = f" {predicted} "
    accepted = [expected for expected in accepted_answers(row) if expected]
    # Whole-token containment, so "3" does not match inside "13".
    return {"exact": predicted in accepted,
            "contains": any(f" {expected} " in padded for expected in accepted)}


def _accuracy(bucket: dict) -> dict:
    scored = bucket["scored"]
    return {**bucket, "exact_accuracy": bucket["exact"] / scored if scored else None,
            "contains_accuracy": bucket["contains"] / scored if scored else None}


def score_predictions(rows: list[dict], predictions) -> dict:
    """Score {id, answer} predictions against holdout rows; missing answers count as wrong."""
    answers = {}
    for prediction in predictions:
        identifier, answer = prediction.get("id"), prediction.get("answer")
        if not isinstance(identifier, str) or not isinstance(answer, str):
            raise ValueError(f"Each prediction needs string id and answer fields, got {prediction!r:.200}")
        if identifier in answers:
            raise ValueError(f"Duplicate prediction for {identifier}")
        answers[identifier] = answer
    known = {row["id"] for row in rows}
    categories, results = {}, []
    total = {"scored": 0, "exact": 0, "contains": 0, "manual_review": 0, "missing": 0}
    for row in rows:
        bucket = categories.setdefault(
            row["category"], {"scored": 0, "exact": 0, "contains": 0, "manual_review": 0, "missing": 0})
        answer = answers.get(row["id"])
        if row["category"] in MANUAL_CATEGORIES:
            flags = None
        else:
            flags = score_answer(row, answer) if answer is not None else {"exact": False, "contains": False}
        results.append({"id": row["id"], "category": row["category"],
                        "missing": answer is None, **(flags or {"manual_review": True})})
        for target in (bucket, total):
            if flags is None:
                target["manual_review"] += 1
                continue
            target["scored"] += 1
            target["exact"] += flags["exact"]
            target["contains"] += flags["contains"]
            target["missing"] += answer is None
    return {"categories": {name: _accuracy(bucket) for name, bucket in categories.items()},
            "total": _accuracy(total), "rows": results,
            "unknown_ids": sorted(set(answers) - known),
            "scope": "Normalized string match only; manual review still decides correctness"}


def load_predictions(path: Path, report_key: str = "simple_questions") -> list[dict]:
    """Read a JSON list, JSONL rows, or a runner report holding a list under report_key."""
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        # A one-line JSONL file parses as a single prediction object rather than a report.
        value = [value] if "id" in value else value.get(report_key, value.get("predictions"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: expected a list of {{id, answer}} objects, JSONL, or a report with {report_key}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSON or JSONL of {id, answer} predictions")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT,
                        help="Question file in the holdout schema, e.g. data/everyday_eval.json")
    parser.add_argument("--json", action="store_true", help="Print the full score report as JSON")
    args = parser.parse_args(argv)
    try:
        rows = json.loads(args.holdout.read_text(encoding="utf-8"))["rows"]
        report_key = REPORT_KEYS.get(args.holdout.name, "simple_questions")
        report = score_predictions(rows, load_predictions(args.predictions, report_key))
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    format_rate = lambda value: "n/a" if value is None else f"{value:.1%}"
    for name, bucket in [*sorted(report["categories"].items()), ("TOTAL", report["total"])]:
        print(f"{name:<14} exact {bucket['exact']}/{bucket['scored']} ({format_rate(bucket['exact_accuracy'])})"
              f"  contains {bucket['contains']}/{bucket['scored']} ({format_rate(bucket['contains_accuracy'])})"
              f"  manual {bucket['manual_review']}  missing {bucket['missing']}")
    if report["unknown_ids"]:
        print(f"ignored unknown ids: {', '.join(report['unknown_ids'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
