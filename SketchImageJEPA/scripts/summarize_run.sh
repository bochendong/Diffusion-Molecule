#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_DIR="${1:-${SKETCHIMAGE_RUN_DIR:-}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: bash scripts/summarize_run.sh outputs/runs/<run_name>" >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.report \
  --predictions "$RUN_DIR/predictions.csv" \
  --out-dir "$RUN_DIR"
