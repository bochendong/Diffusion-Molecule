#!/usr/bin/env bash
# Run offline candidate reranking diagnostics for a saved SketchMolEnd2End run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_E2E_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_DIR="${SKETCHMOL_E2E_RUN_DIR:-SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7}"
PREDICTIONS_CSV="${SKETCHMOL_E2E_PREDICTIONS_CSV:-$RUN_DIR/predictions.csv}"
OUTPUT_DIR="${SKETCHMOL_E2E_RERANK_OUTPUT_DIR:-$RUN_DIR/rerank_diagnostic}"
RERANK_MODES="${SKETCHMOL_E2E_RERANK_MODES:-beam,predicted_fingerprint,render_mse,oracle_tanimoto}"
IMAGE_SIZE="${SKETCHMOL_E2E_IMAGE_SIZE:-128}"
LIMIT="${SKETCHMOL_E2E_RERANK_LIMIT:-}"

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"
export SKETCHMOL_E2E_MODULES="${SKETCHMOL_E2E_MODULES:-gcc rdkit/2025.09.4}"

if [[ -n "${SKETCHMOL_E2E_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_E2E_MODULES
fi

echo "SketchMolEnd2End rerank diagnostic"
echo "  python=$PYTHON_BIN"
echo "  predictions=$PREDICTIONS_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  rerank_modes=$RERANK_MODES"
echo "  image_size=$IMAGE_SIZE"
echo "  limit=${LIMIT:-<all>}"

if [[ ! -f "$PREDICTIONS_CSV" ]]; then
  echo "ERROR: predictions.csv not found: $PREDICTIONS_CSV" >&2
  exit 2
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import rdkit  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  echo "ERROR: rerank diagnostics require RDKit and Pillow." >&2
  exit 2
fi

ARGS=(
  -m sketchmol_end2end.rerank_predictions
  --predictions-csv "$PREDICTIONS_CSV"
  --output-dir "$OUTPUT_DIR"
  --rerank-modes "$RERANK_MODES"
  --image-size "$IMAGE_SIZE"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Rerank diagnostic finished: $OUTPUT_DIR"
echo "  summary=$OUTPUT_DIR/rerank_summary.json"
echo "  summary_csv=$OUTPUT_DIR/rerank_summary.csv"
