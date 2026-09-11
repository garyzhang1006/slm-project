"""Measure transfer after expanding the short-answer pilot's training corpus."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_short_qa_pilot import main


if __name__ == "__main__":
    main(broad=True)
