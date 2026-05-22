#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -m sketchimage_jepa.experiment \
  --output-dir outputs/smoke \
  --feature-dim 96 \
  --latent-dim 48 \
  --top-k 5 \
  --train-fraction 0.67 \
  --seed 7 \
  --render-image-context
