#!/usr/bin/env python3
"""Validate and copy a locally sourced JSONL file into canonical project format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_slm.data import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", dest="license_name", required=True)
    args = parser.parse_args()
    examples = load_jsonl(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = example.to_dict()
            record["source"] = args.source
            record["license"] = args.license_name
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"validated and wrote {len(examples)} records to {output}")


if __name__ == "__main__":
    main()
