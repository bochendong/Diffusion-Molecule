#!/usr/bin/env bash
# Sweep non-oracle reranking weights for one completed run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_DIR="${1:-${SKETCHIMAGE_RUN_DIR:-}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: bash scripts/rerank_run.sh outputs/runs/<run_name>" >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.rerank_predictions \
  --predictions "$RUN_DIR/predictions.csv" \
  --out-dir "$RUN_DIR/rerank_diagnostics" \
  --base-weights "${SKETCHIMAGE_RERANK_BASE_WEIGHTS:-0,0.25,0.5,0.75,1.0}" \
  --source-weights "${SKETCHIMAGE_RERANK_SOURCE_WEIGHTS:-0,0.15,0.35,0.55}" \
  --property-weights "${SKETCHIMAGE_RERANK_PROPERTY_WEIGHTS:-0,0.10,0.25,0.40}" \
  --scaffold-weights "${SKETCHIMAGE_RERANK_SCAFFOLD_WEIGHTS:-0,0.10,0.20,0.30}" \
  --property-delta-weights "${SKETCHIMAGE_RERANK_PROPERTY_DELTA_WEIGHTS:-0,0.25,0.50,0.75,1.0,2.0,4.0,8.0}"
