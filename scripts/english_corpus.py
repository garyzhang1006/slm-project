"""Pure adapters for licensed English corpora; downloading belongs to the runner."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable

from cognition_slm.data import DataValidationError, format_prompt, validate_record
from cognition_slm.tokenizer import ByteTokenizer


def _bounded_record(
    record_id: str, prompt: str, answer: str, source: str, license_name: str,
    max_bytes: int,
) -> dict[str, Any] | None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer token budget")
    raw = {
        "id": record_id,
        "prompt": prompt,
        "answer": answer,
        "task_type": "language_generation",
        "confidence": 0.9,
        "error_category": "none",
        "source": source,
        "license": license_name,
    }
    try:
        example = validate_record(raw)
    except DataValidationError:
        return None
    # Count the actual serialized training text, including BOS and EOS.
    tokens = ByteTokenizer().encode(format_prompt(example) + example.answer.strip())
    return example.to_dict() if len(tokens) <= max_bytes else None


def story_record(
    text: str, source_id: str, max_bytes: int = 768,
) -> dict[str, Any] | None:
    """Use a complete opening sentence to request the untouched story remainder."""
    if not isinstance(text, str) or not isinstance(source_id, str) or not source_id.strip():
        return None
    text = text.strip()
    boundary = re.search(r"[.!?][\"'\u201d\u2019]*(?=\s|$)", text)
    if boundary is None:
        return None
    opening = text[:boundary.end()]
    if not 20 <= len(opening.encode("utf-8")) <= 180:
        return None
    answer = text[boundary.end():].lstrip()
    return _bounded_record(
        f"tinystories-{source_id}",
        f"Continue this story in English:\n{opening}",
        answer, "roneneldan/TinyStories", "cdla-sharing-1.0", max_bytes,
    )


def dolly_record(
    raw: dict[str, Any], index: int, max_bytes: int = 768,
) -> dict[str, Any] | None:
    """Preserve Dolly's instruction, optional supporting context, and response."""
    if not isinstance(raw, dict):
        return None
    instruction = raw.get("instruction")
    context = raw.get("context", "")
    response = raw.get("response")
    if not all(isinstance(value, str) for value in (instruction, context, response)):
        return None
    if not instruction.strip():
        return None
    prompt = instruction
    if context.strip():
        prompt += f"\n\nContext:\n{context}"
    return _bounded_record(
        f"dolly-{index}", prompt, response,
        "databricks/databricks-dolly-15k", "CC-BY-SA-3.0", max_bytes,
    )


def _normalized_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def deterministic_split(
    records: Iterable[dict[str, Any]], eval_count: int = 128,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate prompts and hold out a hash-ordered, input-order-stable subset.

    Conflicting responses to the same normalized prompt retain one deterministic
    representative. Callers must check that enough training records remain.
    """
    if isinstance(eval_count, bool) or not isinstance(eval_count, int) or eval_count < 0:
        raise ValueError("eval_count must be a non-negative integer")
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw in records:
        record = validate_record(raw).to_dict()
        prompt = _normalized_prompt(record["prompt"])
        serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
        previous = unique.get(prompt)
        if previous is None or serialized < previous[0]:
            unique[prompt] = (serialized, record)
    prompts = sorted(
        unique,
        key=lambda prompt: (hashlib.sha256(prompt.encode("utf-8")).hexdigest(), prompt),
    )
    evaluation = [unique[prompt][1] for prompt in prompts[:eval_count]]
    training = [unique[prompt][1] for prompt in prompts[eval_count:]]
    return training, evaluation
