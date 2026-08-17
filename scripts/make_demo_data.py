#!/usr/bin/env python3
"""Regenerate project-authored demo data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognition_slm.demo_data import EVAL_ROWS, TRAIN_ROWS, write_jsonl


def main() -> None:
    write_jsonl(TRAIN_ROWS, ROOT / "data" / "demo.jsonl")
    write_jsonl(EVAL_ROWS, ROOT / "data" / "eval.jsonl")
    print(f"wrote {len(TRAIN_ROWS)} training and {len(EVAL_ROWS)} eval records")


if __name__ == "__main__":
    main()
