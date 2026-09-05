#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv-ui/bin/python ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv or create .venv-ui with Python 3.11 and install this project first."
    exit 1
  fi
  uv venv --python 3.11 .venv-ui
fi
if ! .venv-ui/bin/python -c 'import torch, numpy' >/dev/null 2>&1; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Studio dependencies missing. Install uv and rerun this launcher."
    exit 1
  fi
  uv pip install --python .venv-ui/bin/python -e .
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting SLM Studio"
echo "Keep this terminal open. Press Control-C to stop."
exec .venv-ui/bin/python -m cognition_slm.server "$@"
