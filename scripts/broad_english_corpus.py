"""Pure adapters for complete English paragraphs from FineWeb-Edu."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from cognition_slm.audit import SECRET_PATTERNS
from scripts.english_corpus import _bounded_record


def paragraph_records(raw: dict, max_bytes: int = 1024) -> list[dict[str, Any]]:
    """Keep up to three complete, bounded paragraphs with their provenance."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer token budget")
    if not isinstance(raw, dict):
        return []
    fields = [raw.get(key) for key in ("id", "text", "url", "language")]
    if not all(isinstance(value, str) and value.strip() for value in fields):
        return []
    document_id, text, url, language = fields
    if language.strip().casefold() != "en":
        return []
    if any(pattern.search(value) for pattern in SECRET_PATTERNS for value in fields):
        return []
    try:
        document_id.encode("utf-8")
        text.encode("utf-8")
        url.encode("utf-8")
    except UnicodeEncodeError:
        return []

    # Preserve wrapped lines when blank lines explicitly separate paragraphs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    separator = r"\n[ \t]*\n+" if re.search(r"\n[ \t]*\n", text) else r"\n"
    records = []
    for index, paragraph in enumerate(re.split(separator, text)):
        paragraph = paragraph.strip()
        if len(paragraph.encode("utf-8")) < 200:
            continue
        if re.search(r"[.!?][\"'\u201d\u2019]*$", paragraph) is None:
            continue
        boundary = re.search(r"[.!?][\"'\u201d\u2019]*(?=\s|$)", paragraph)
        if boundary is None:
            continue
        opening = paragraph[:boundary.end()]
        if not 20 <= len(opening.encode("utf-8")) <= 180:
            continue
        answer = paragraph[boundary.end():].lstrip()
        digest = hashlib.sha256(f"{document_id}\0{index}".encode("utf-8")).hexdigest()
        record = _bounded_record(
            f"fineweb-edu-{digest}",
            f"Continue this English passage:\n{opening}",
            answer,
            f"HuggingFaceFW/fineweb-edu; {url}",
            "ODC-BY-1.0 (database); underlying text rights retained",
            max_bytes,
        )
        if record is not None:
            records.append(record)
            if len(records) == 3:
                break
    return records
