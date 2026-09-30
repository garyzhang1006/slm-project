#!/usr/bin/env python3
"""Validate and copy a locally sourced JSONL file into canonical project format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# load_jsonl rejects a record without provenance, so this reads through the same parser and fills gaps first.
from cognition_slm.data import DataValidationError, _parse_json_line, _utf8_lines, validate_record


def load_with_provenance(path: Path, source: str, license_name: str) -> list[dict]:
    """Validate records like load_jsonl, filling source and license only where a record has none."""
    if not path.exists():
        raise FileNotFoundError(path)
    records, seen_ids = [], set()
    for record_number, line in _utf8_lines(path):
        if not line.strip():
            continue
        raw = _parse_json_line(path, record_number, line)
        if isinstance(raw, dict):
            for field, default in (("source", source), ("license", license_name)):
                value = raw.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    raw[field] = default
        example = validate_record(raw, record_number)
        if example.id in seen_ids:
            raise DataValidationError(f"{path}:{record_number}: duplicate id {example.id!r}")
        seen_ids.add(example.id)
        records.append(example.to_dict())
    if not records:
        raise DataValidationError(f"{path}: dataset has no records")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True, help="Source for records that do not name their own")
    parser.add_argument("--license", dest="license_name", required=True,
                        help="License for records that do not name their own")
    args = parser.parse_args(argv)
    records = load_with_provenance(Path(args.input), args.source, args.license_name)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"validated and wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
