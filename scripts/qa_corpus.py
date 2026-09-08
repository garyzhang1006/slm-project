"""Passage-grounded question answering records and SQuAD-style answer metrics."""

from __future__ import annotations

from collections import Counter
import math
import re
import string
from typing import Any, Iterable

from scripts.english_corpus import _bounded_record


def squad_record(raw: dict[str, Any], max_bytes: int = 1024) -> dict[str, Any] | None:
    """Keep the full passage and a verified short reference answer, or reject it."""
    if not isinstance(raw, dict):
        return None
    record_id, context, question = (raw.get(key) for key in ("id", "context", "question"))
    if not all(isinstance(value, str) and value.strip()
               for value in (record_id, context, question)):
        return None
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        return None
    texts, starts = answers.get("text"), answers.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list) or not texts or not starts:
        return None
    if len(texts) != len(starts):
        return None
    answer, start = texts[0], starts[0]
    if not isinstance(answer, str) or not answer.strip() or len(answer.encode("utf-8")) > 96:
        return None
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        return None
    # SQuAD offsets refer to characters in the original passage, not UTF-8 bytes.
    if context[start:start + len(answer)] != answer:
        return None
    prompt = (
        "Answer the question using the passage. Reply with only the answer."
        f"\n\nPassage: {context}\n\nQuestion: {question}"
    )
    return _bounded_record(
        f"squad-{record_id}", prompt, answer, "rajpurkar/squad", "CC-BY-SA-4.0", max_bytes,
    )


def normalized_answer(text: str) -> str:
    """Apply SQuAD's lowercase, ASCII punctuation, article, and whitespace rules."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, references: Iterable[str]) -> dict[str, float]:
    """Return the best exact-match and multiset token-F1 scores over references."""
    normalized = normalized_answer(prediction)
    prediction_tokens = normalized.split()
    exact_match = token_f1 = 0.0
    for reference in references:
        reference_normalized = normalized_answer(reference)
        reference_tokens = reference_normalized.split()
        exact_match = max(exact_match, float(normalized == reference_normalized))
        if not prediction_tokens or not reference_tokens:
            score = float(prediction_tokens == reference_tokens)
        else:
            overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
            score = 2.0 * overlap / (len(prediction_tokens) + len(reference_tokens))
        token_f1 = max(token_f1, score)
    return {"exact_match": exact_match, "token_f1": token_f1}


def pilot_improved(baseline: dict[str, float], candidate: dict[str, float]) -> bool:
    """Continue training only after a measurable held-out answer-quality gain."""
    for label, metrics in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(metrics, dict):
            raise ValueError(f"{label} must contain exact_match and token_f1 metrics")
        for key in ("exact_match", "token_f1"):
            value = metrics.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{label}.{key} must be a finite number between 0 and 1")
    return (
        candidate["exact_match"] >= baseline["exact_match"] + 0.02
        or (candidate["token_f1"] >= baseline["token_f1"] + 0.05
            and candidate["exact_match"] >= baseline["exact_match"])
    )
