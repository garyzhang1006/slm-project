#!/bin/bash
set -euo pipefail
# Model paths the user typed are relative to where they ran this, not to the project folder it moves into.
CALLER_DIR="$(pwd -P)"
ARGS=()
EXPECT_PATH=false
for arg in "$@"; do
  if [ "$EXPECT_PATH" = true ] && [ -n "$arg" ] && [ "${arg#/}" = "$arg" ]; then
    arg="$CALLER_DIR/$arg"
  fi
  EXPECT_PATH=false
  case "$arg" in
    --checkpoint|--lora-adapter) EXPECT_PATH=true ;;
    --checkpoint=?*|--lora-adapter=?*)
      value="${arg#*=}"
      if [ "${value#/}" = "$value" ]; then arg="${arg%%=*}=$CALLER_DIR/$value"; fi ;;
  esac
  ARGS+=("$arg")
done
cd "$(dirname "$0")"
VENV_DIR=".venv-ui-py313"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv or create $VENV_DIR with Python 3.13 and install this project first."
    exit 1
  fi
  uv venv --python 3.13 "$VENV_DIR"
fi
SOURCES_ONLY=false
for arg in "$@"; do
  if [ "$arg" = "--sources-only" ]; then
    SOURCES_ONLY=true
  fi
done
# Reference retrieval uses the standard library and must not depend on PyTorch.
if [ "$SOURCES_ONLY" = false ]; then
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
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Starting slm studio. It opens in your browser once the server is up."
echo "Keep this window open while you use it, and press Control-C here to stop it."
# ${ARGS[@]+...} stops bash 3.2, the macOS default, from treating an empty array as unbound under set -u.
exec "$VENV_DIR/bin/python" -m cognition_slm.server --open ${ARGS[@]+"${ARGS[@]}"}
