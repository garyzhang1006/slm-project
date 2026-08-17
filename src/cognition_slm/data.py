"""Canonical JSONL data schema and encoding helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ERROR_CATEGORIES, TASK_TYPES
from .tokenizer import ByteTokenizer


class DataValidationError(ValueError):
    """Raised when a record violates the project data contract."""


DISALLOWED_FIELDS = {
    "chain_of_thought",
    "cot",
    "hidden_reasoning",
    "private_thoughts",
    "internal_monologue",
}
ALLOWED_FIELDS = {
    "id",
    "prompt",
    "answer",
    "task_type",
    "confidence",
    "error_category",
    "source",
    "license",
}
MAX_TEXT_CHARS = 100_000


@dataclass(frozen=True)
class CognitionExample:
    id: str
    prompt: str
    answer: str
    task_type: str
    confidence: float
    error_category: str
    source: str
    license: str

    @property
    def confidence_bucket(self) -> int:
        if self.confidence < 0.4:
            return 0
        if self.confidence < 0.7:
            return 1
        return 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "answer": self.answer,
            "task_type": self.task_type,
            "confidence": self.confidence,
            "error_category": self.error_category,
            "source": self.source,
            "license": self.license,
        }


def _required_text(raw: dict[str, Any], field: str, record_number: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"record {record_number}: {field} must be non-empty text")
    if len(value) > MAX_TEXT_CHARS:
        raise DataValidationError(f"record {record_number}: {field} exceeds {MAX_TEXT_CHARS} characters")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise DataValidationError(f"record {record_number}: {field} contains a control character")
    return value


def validate_record(raw: dict[str, Any], record_number: int = 0) -> CognitionExample:
    if not isinstance(raw, dict):
        raise DataValidationError(f"record {record_number}: expected JSON object")
    disallowed = sorted(DISALLOWED_FIELDS.intersection(raw))
    if disallowed:
        raise DataValidationError(
            f"record {record_number}: disallowed hidden-reasoning fields: {', '.join(disallowed)}"
        )
    unknown = sorted(set(raw).difference(ALLOWED_FIELDS))
    if unknown:
        raise DataValidationError(f"record {record_number}: unknown fields: {', '.join(unknown)}")
    record_id = _required_text(raw, "id", record_number)
    prompt = _required_text(raw, "prompt", record_number)
    answer = _required_text(raw, "answer", record_number)
    task_type = _required_text(raw, "task_type", record_number)
    if task_type not in TASK_TYPES:
        raise DataValidationError(f"record {record_number}: unknown task_type {task_type!r}")
    error_category = _required_text(raw, "error_category", record_number)
    if error_category not in ERROR_CATEGORIES:
        raise DataValidationError(
            f"record {record_number}: unknown error_category {error_category!r}"
        )
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise DataValidationError(f"record {record_number}: confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise DataValidationError(f"record {record_number}: confidence must be between 0 and 1")
    source = _required_text(raw, "source", record_number)
    license_name = _required_text(raw, "license", record_number)
    return CognitionExample(
        id=record_id,
        prompt=prompt,
        answer=answer,
        task_type=task_type,
        confidence=float(confidence),
        error_category=error_category,
        source=source,
        license=license_name,
    )


def load_jsonl(path: str | Path) -> list[CognitionExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    examples: list[CognitionExample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for record_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
            except (DataValidationError, json.JSONDecodeError) as exc:
                detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise DataValidationError(f"{path}:{record_number}: invalid JSON: {detail}") from exc
            example = validate_record(raw, record_number)
            if example.id in seen_ids:
                raise DataValidationError(f"{path}:{record_number}: duplicate id {example.id!r}")
            seen_ids.add(example.id)
            examples.append(example)
    if not examples:
        raise DataValidationError(f"{path}: dataset has no records")
    return examples


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DataValidationError(f"non-finite JSON number {value}")


def format_prompt(example: CognitionExample) -> str:
    return (
        f"<task_type>{example.task_type}</task_type>\n"
        f"<instruction>\n{example.prompt.strip()}\n</instruction>\n"
        "<answer>\n"
    )


def format_training_text(example: CognitionExample) -> str:
    return format_prompt(example) + example.answer.strip()


def encode_examples(
    examples: Iterable[CognitionExample], tokenizer: ByteTokenizer, block_size: int
) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for example in examples:
        encoded.append(
            {
                "id": example.id,
                "input_ids": tokenizer.encode(
                    format_training_text(example), max_length=block_size
                ),
                "task_label": TASK_TYPES.index(example.task_type),
                "error_label": ERROR_CATEGORIES.index(example.error_category),
                "confidence_label": example.confidence_bucket,
            }
        )
    return encoded
