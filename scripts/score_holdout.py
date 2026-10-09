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
# Where the word form is what the question tests (plural of mouse, is or are, "write only the word window"),
# an inflected answer is wrong. So it is for opposites: "evens" is not the opposite of odd, nor "answered" of question.
# Reading answers are words from the passage and list any other form they take, so "icing" does not answer "ice".
INFLECTION_EXEMPT = frozenset({"plurals", "english", "instruction", "opposites", "reading"})
_SUFFIXES = ("s", "es", "ed", "d", "ing")
_UNITS = {word: index for index, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {word: 10 * index for index, word in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
# A minus sign counts only when no word or digit comes right before it, so "3-4" stays two numbers.
_TOKENS = re.compile(r"(?:(?<!\w)-)?(?:\d+(?:\.\d+)?|(?<![\w.])\.\d+)|[^\W\d_]+")
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
# Subscript digits spell the same number, so CO₂ reads as CO2. Superscripts stay, since 5² is not 52.
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_MINUS = str.maketrans({"\u2212": "-", "\u2013": "-", "\uff0d": "-"})
# A digit may touch the time on either side, as in "2p.m." and "2:00pm".
_MERIDIEM = re.compile(r"(?<![^\W\d_])([ap])\. ?m\b\.?")
_CLOCK = re.compile(r"\b(\d{1,2}):00(?!\d)")
# Spelled numbers continue only across spaces and hyphens, so "One hundred. Ten decades" stays two numbers.
_JOINER = re.compile(r"[\s-]*")
# An abstain reply is exact when every clause between punctuation, a spaced dash or a spaced "but", after
# one leading and, so or but, is an accepted refusal or an opener below followed by an object of at most
# five words that starts with your, about or a question word, has no answer word before its last word and
# no number but one. Is, it, its or was right after the question word belongs to the question being declined.
# So "I don't know who your best friend is" and "I don't know what is inside the box" are exact, and
# "I don't know your name, it's Sam.", "I don't know your name is Sam." and "I don't know about 50." are not.
_CLAUSES = re.compile("[.,;:!?()\u2013\u2014]+|\\s-+\\s|\\sbut\\s", re.IGNORECASE)
_REFUSAL_OPENERS = ("i dont know", "i do not know", "you havent told me", "you have not told me")
_QUESTION_WORDS = frozenset({"what", "where", "who", "when", "which", "how"})
_REFUSAL_OBJECTS = _QUESTION_WORDS | {"your", "about"}
_REFUSAL_OBJECT_WORDS = 5
_ANSWER_WORDS = frozenset({"i", "is", "was", "it", "its", "maybe", "probably"})
_QUESTION_FOLLOWERS = frozenset({"is", "it", "its", "was"})
_CLAUSE_JOINERS = frozenset({"and", "so", "but"})


def _decimal(token: str) -> str:
    """"8.00" as 8 and ".5" as 0.5, so trailing zeros and a missing leading zero keep the number's value."""
    whole, _, fraction = token.partition(".")
    fraction = fraction.rstrip("0")
    if whole in ("", "-"):
        whole += "0"
    return f"{whole}.{fraction}" if fraction else whole


def _below_100(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Value and next index of "seven", "twelve", "forty" or "forty two" at tokens[i]."""
    if i < len(tokens) and tokens[i] in _UNITS:
        return _UNITS[tokens[i]], i + 1
    if i < len(tokens) and tokens[i] in _TENS:
        # "twenty one" (or "twenty-one", split on the hyphen) is one number; "three four" stays two.
        if i + 1 < len(tokens) and 0 < _UNITS.get(tokens[i + 1], 0) < 10:
            return _TENS[tokens[i]] + _UNITS[tokens[i + 1]], i + 2
        return _TENS[tokens[i]], i + 1
    return None


def _below_1000(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Like _below_100, plus "a hundred" and "one hundred and forty four"."""
    if i + 1 < len(tokens) and tokens[i + 1] == "hundred" and (tokens[i] == "a" or 0 < _UNITS.get(tokens[i], 0) < 10):
        value, i = 100 * _UNITS.get(tokens[i], 1), i + 2
        rest = _below_100(tokens, i + (i < len(tokens) and tokens[i] == "and"))
        return (value + rest[0], rest[1]) if rest and rest[0] else (value, i)
    return _below_100(tokens, i)


def _number(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Value and next index of a spelled number below one million at tokens[i], or None."""
    first = _below_1000(tokens, i) or ((1, i + 1) if tokens[i] == "a" else None)
    if first is None:
        return None
    value, i = first
    if i < len(tokens) and tokens[i] == "thousand" and value:
        value, i = 1000 * value, i + 1
        rest = _below_1000(tokens, i + (i < len(tokens) and tokens[i] == "and"))
        return (value + rest[0], rest[1]) if rest and rest[0] else (value, i)
    # A bare "a" is an article; it counts as one only before "hundred" or "thousand".
    return None if tokens[i - 1] == "a" else (value, i)


def normalize_answer(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace and write spelled numbers below a million as digits."""
    text = _GROUPED.sub(lambda match: match.group().replace(",", ""), text)  # "1,000" is one number
    # U+2212, an en dash and a fullwidth hyphen-minus also write a minus sign, so "−25" must not read as 25.
    text = text.casefold().replace("'", "").replace("’", "").translate(_MINUS).translate(_SUBSCRIPTS)
    # "2 p.m." and "2:00 pm" say what "2 pm" says; other minutes stay, so "2:30" is not 2.
    text = _CLOCK.sub(r"\1", _MERIDIEM.sub(r"\1m", text))
    phrases, end = [], None
    for match in _TOKENS.finditer(text):
        if end is None or not _JOINER.fullmatch(text, end, match.start()):
            phrases.append([])
        phrases[-1].append(_decimal(match.group()) if "." in match.group() else match.group())
        end = match.end()
    result = []
    for tokens in phrases:
        i = 0
        while i < len(tokens):
            number = _number(tokens, i)
            if number is None:
                result.append(tokens[i])
                i += 1
            else:
                result.append(str(number[0]))
                i = number[1]
    return " ".join(result)


def accepted_answers(row: dict) -> list[str]:
    """Rows may list accepted_answers; the current holdout only has expected_rubric."""
    answers = row.get("accepted_answers") or [row["expected_rubric"]]
    return [normalize_answer(answer) for answer in answers]


def inflections(word: str) -> set[str]:
    """Regular inflections of a one-word answer, so "A horse neighs" or "whinnies" still matches."""
    forms = {word + suffix for suffix in _SUFFIXES}
    if word.endswith("e"):
        forms.add(word[:-1] + "ing")
    if word.endswith("y"):
        forms |= {word[:-1] + "ies", word[:-1] + "ied"}
    return forms


def _is_refusal(clause: str, accepted: list[str]) -> bool:
    """Whether a normalized clause of an abstain reply only refuses, by the rule above _CLAUSES."""
    words = clause.split()
    if words and words[0] in _CLAUSE_JOINERS:
        words = words[1:]
    if " ".join(words) in accepted:
        return True
    for opener in map(str.split, _REFUSAL_OPENERS):
        tail = words[len(opener):]
        if (words[:len(opener)] != opener or not 0 < len(tail) <= _REFUSAL_OBJECT_WORDS
                or tail[0] not in _REFUSAL_OBJECTS):
            continue
        declined = len(tail) > 1 and tail[0] in _QUESTION_WORDS and tail[1] in _QUESTION_FOLLOWERS
        checked = tail[2:] if declined else tail
        # A number is a guess, as in "about 50"; normalize_answer writes the pronoun in "which one" as 1.
        guess = any(word.lstrip("-")[:1].isdigit() and word != "1" for word in tail)
        if not guess and not _ANSWER_WORDS.intersection(checked[:-1]):
            return True
    return False


def score_answer(row: dict, answer: str) -> dict | None:
    """Return exact/contains flags, or None when the row needs manual review."""
    if row["category"] in MANUAL_CATEGORIES:
        return None
    predicted = normalize_answer(answer)
    padded = f" {predicted} "
    accepted = [expected for expected in accepted_answers(row) if expected]
    # A row whose inflected answer names something else ("Germans" are people, not a language) sets "inflect": false.
    if row["category"] not in INFLECTION_EXEMPT and row.get("inflect", True):
        # Only alphabetic single words of 3+ letters: "no" -> "nos" or "7" -> "7s" would add noise, not recall.
        accepted += sorted({form for expected in accepted if expected.isalpha() and len(expected) >= 3
                            for form in inflections(expected)})
    # A reply made only of accepted answers, such as "I don't know. You haven't told me.", says nothing else.
    sentences = [normalize_answer(part) for part in re.split(r"(?<=[.!?])\s+", answer.strip())]
    exact = predicted in accepted or (len(sentences) > 1 and all(sentence in accepted for sentence in sentences))
    if row["category"] == "abstain" and not exact:
        clauses = [clause for clause in map(normalize_answer, _CLAUSES.split(answer)) if clause]
        exact = bool(clauses) and all(_is_refusal(clause, accepted) for clause in clauses)
    # Whole-token containment, so "3" does not match inside "13".
    return {"exact": exact,
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
            target["missing"] += answer is None
            if flags is None:
                target["manual_review"] += 1
                continue
            target["scored"] += 1
            target["exact"] += flags["exact"]
            target["contains"] += flags["contains"]
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
        # splitlines() would also break on U+2028, which json.dumps(ensure_ascii=False) leaves raw in answers.
        value = [json.loads(line) for line in text.split("\n") if line.strip()]
    if isinstance(value, dict):
        # A one-line JSONL file parses as a single prediction object rather than a report.
        # kaggle_simple_questions_audit.py keeps its answered holdout rows under "rows".
        value = [value] if "id" in value else value.get(report_key, value.get("predictions", value.get("rows")))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: expected a list of {{id, answer}} objects, JSONL, or a report with {report_key}")
    if not value:
        raise ValueError(f"{path}: no predictions found; check the file is the runner's finished output")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSON or JSONL of {id, answer} predictions")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT,
                        help="Question file in the holdout schema, e.g. data/everyday_eval.json")
    parser.add_argument("--json", action="store_true", help="Print the full score report as JSON")
    args = parser.parse_args(argv)
    try:
        holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
        rows = holdout.get("rows") if isinstance(holdout, dict) else None
        # A swapped argument order passes a predictions list here, which would crash with a TypeError.
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{args.holdout}: expected a question file with a rows list, "
                             "like data/simple_questions_holdout.json")
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
