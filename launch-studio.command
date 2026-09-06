#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
VENV_DIR=".venv-ui-py313"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv or create $VENV_DIR with Python 3.13 and install this project first."
    exit 1
  fi
  uv venv --python 3.13 "$VENV_DIR"
fi
if ! "$VENV_DIR/bin/python" -c 'import importlib.util; raise SystemExit(any(importlib.util.find_spec(name) is None for name in ("torch", "numpy")))'; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Studio dependencies missing. Install uv and rerun this launcher."
    exit 1
  fi
  uv pip install --python "$VENV_DIR/bin/python" -e .
fi
# A stuck native import must fail with a deadline instead of hanging the launcher.
"$VENV_DIR/bin/python" -c '
import subprocess, sys
try:
    result = subprocess.run([sys.executable, "-c", "import torch, numpy"], timeout=30)
except subprocess.TimeoutExpired:
    sys.exit("PyTorch import timed out after 30 seconds. Studio was not started; check the Python environment.")
if result.returncode:
    sys.exit("PyTorch import failed. Repair the Studio environment before retrying.")
'
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting SLM Studio"
echo "Keep this terminal open. Press Control-C to stop."
exec "$VENV_DIR/bin/python" -m cognition_slm.server "$@"
