#!/usr/bin/env bash
# Summarize a paper-track ablation matrix across variants and seeds.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${SKETCHIMAGE_PAPER_MODE:-pilot}"
if [[ -n "${SKETCHIMAGE_PAPER_MATRIX_NAME:-}" ]]; then
  MATRIX_NAME="$SKETCHIMAGE_PAPER_MATRIX_NAME"
elif [[ "$MODE" == "full" ]]; then
  MATRIX_NAME="sketchmol_aligned_paper_full"
else
  MATRIX_NAME="sketchmol_aligned_paper_pilot"
fi

if [[ -n "${SKETCHIMAGE_PAPER_SEEDS:-}" ]]; then
  SEEDS="$SKETCHIMAGE_PAPER_SEEDS"
elif [[ "$MODE" == "full" ]]; then
  SEEDS="7 13 23"
else
  SEEDS="7"
fi

if [[ -n "${SKETCHIMAGE_PAPER_VARIANTS:-}" ]]; then
  VARIANTS="$SKETCHIMAGE_PAPER_VARIANTS"
elif [[ "$MODE" == "full" ]]; then
  VARIANTS="ridge_baseline planner_best no_contrastive weak_contrastive no_image_context"
else
  VARIANTS="ridge_baseline planner_best no_contrastive no_image_context"
fi

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUT_CSV="${SKETCHIMAGE_PAPER_OUT_CSV:-outputs/paper/${MATRIX_NAME}_summary.csv}"
OUT_JSON="${SKETCHIMAGE_PAPER_OUT_JSON:-outputs/paper/${MATRIX_NAME}_summary.json}"

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.paper_matrix \
  --matrix-name "$MATRIX_NAME" \
  --variants "$VARIANTS" \
  --seeds "$SEEDS" \
  --out-csv "$OUT_CSV" \
  --out-json "$OUT_JSON"
