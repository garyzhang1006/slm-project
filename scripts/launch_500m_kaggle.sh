#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="${1:-}"
SLUG="${2:-slm-500m-english-code-quality}"
OUTPUT_DIR="${3:-/tmp/slm-kaggle-500m-quality}"

if [[ -z "$OWNER" ]]; then
  echo "Usage: $0 KAGGLE_OWNER [KERNEL_SLUG] [OUTPUT_DIR]" >&2
  exit 2
fi
if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install and authenticate it before running this launcher." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to package the reviewed source." >&2
  exit 1
fi

python3 "$ROOT_DIR/scripts/prepare_kaggle.py" \
  --owner "$OWNER" \
  --slug "$SLUG" \
  --runner kaggle_500m_quality_run.py \
  --out "$OUTPUT_DIR"
kaggle kernels push -p "$OUTPUT_DIR" --accelerator NvidiaTeslaT4
echo "Submitted Kaggle kernel: $OWNER/$SLUG"
