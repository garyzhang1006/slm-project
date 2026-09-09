"""Kaggle entrypoint for tests, 500M cache timing, and optimizer-resume smoke."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_efficient_run import main

if __name__ == "__main__":
    main(verify_only=True)
