"""Canonical JSONL data schema and encoding helpers."""

from __future__ import annotations

import hashlib
import json
import re
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
# A form feed, or a newline followed by one or more whitespace-only lines, ends a .txt document.
TEXT_DOCUMENT_BREAK = re.compile(r"\f|\r?\n(?:[ \t\r\v]*\n)+")


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
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise DataValidationError(f"record {record_number}: {field} contains a lone surrogate, which is not valid UTF-8") from None
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
    try:
        in_range = 0.0 <= float(confidence) <= 1.0
    except OverflowError:
        in_range = False
    if not in_range:
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


def encode_prompt(
    example: CognitionExample, tokenizer: ByteTokenizer, block_size: int
) -> list[int]:
    """Encode only prompt tokens so auxiliary heads cannot read the answer."""
    return tokenizer.encode(format_prompt(example), add_eos=False, max_length=block_size)


def encode_examples(
    examples: Iterable[CognitionExample], tokenizer: ByteTokenizer, block_size: int, *,
    task_types: tuple[str, ...] = TASK_TYPES, error_categories: tuple[str, ...] = ERROR_CATEGORIES,
) -> list[dict[str, Any]]:
    """Label with the model's own classes; a resumed checkpoint may predate newer task types."""
    encoded: list[dict[str, Any]] = []
    for example in examples:
        for field, value, labels in (("task_type", example.task_type, task_types),
                                     ("error_category", example.error_category, error_categories)):
            if value not in labels:
                raise DataValidationError(
                    f"example {example.id!r} has {field} {value!r}, which this model was not built with; "
                    "drop those records or train a new model"
                )
        prompt_ids = tokenizer.encode(format_prompt(example), add_eos=False)
        full_ids = tokenizer.encode(format_training_text(example))
        # A cut answer is a continuation, so do not teach EOS at an artificial boundary.
        input_ids = full_ids[:block_size]
        pool_position = min(len(prompt_ids), len(input_ids)) - 1
        if pool_position < 0:
            raise ValueError(f"prompt for {example.id!r} produced no tokens")
        answer_start = min(len(prompt_ids), len(input_ids))
        if answer_start >= len(input_ids):
            raise ValueError(
                f"example {example.id!r} has no answer tokens within block_size {block_size}"
            )
        encoded.append(
            {
                "id": example.id,
                "original_tokens": len(full_ids),
                "truncated": len(full_ids) > block_size,
                "input_ids": input_ids,
                "pool_position": pool_position,
                "answer_start": answer_start,
                "task_label": task_types.index(example.task_type),
                "error_label": error_categories.index(example.error_category),
                "confidence_label": example.confidence_bucket,
            }
        )
    return encoded


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrain_text(path: str | Path) -> list[str]:
    """Read raw documents: a .jsonl file with a "text" field per line, else a text file split on
    TEXT_DOCUMENT_BREAK (blank lines or form feeds); a file without breaks stays one document."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".jsonl":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise DataValidationError(f"{path}: text file is empty")
        if not TEXT_DOCUMENT_BREAK.search(text):
            return [text]
        return [document for document in TEXT_DOCUMENT_BREAK.split(text) if document.strip()]
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for record_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
            except (DataValidationError, json.JSONDecodeError) as exc:
                detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise DataValidationError(f"{path}:{record_number}: invalid JSON: {detail}") from exc
            text = raw.get("text") if isinstance(raw, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise DataValidationError(f"{path}:{record_number}: text must be non-empty text")
            documents.append(text)
    if not documents:
        raise DataValidationError(f"{path}: dataset has no records")
    return documents


def pack_pretrain_text(
    documents: Iterable[str], tokenizer: ByteTokenizer, block_size: int
) -> list[dict[str, Any]]:
    """Concatenate BOS/EOS-delimited documents into block_size rows with loss on every target.

    Rows carry no auxiliary labels, so training skips the task/error/confidence heads.
    Only the final row may be shorter; it is kept so small eval files still yield a row.
    """
    stream: list[int] = []
    for document in documents:
        stream.extend(tokenizer.encode(document))
    rows = [stream[start:start + block_size] for start in range(0, len(stream), block_size)]
    if rows and len(rows[-1]) < 2:
        rows.pop()
    if not rows:
        raise ValueError("pretraining text produced no rows with at least two tokens")
    return [
        {
            "id": f"packed-{index}",
            "input_ids": row,
            "pool_position": len(row) - 1,
            # Target positions start at 1, so every next-token prediction is supervised.
            "answer_start": 1,
            "task_label": None,
            "error_label": None,
            "confidence_label": None,
        }
        for index, row in enumerate(rows)
    ]
