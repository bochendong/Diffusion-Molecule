#!/usr/bin/env bash
# Run Phase 5A-0 oracle paired-output baseline.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
RUN_NAME="${SKETCHSMILES_RUN_NAME:-phase5a0_oracle_baseline_seed${SKETCHSMILES_SEED:-7}}"
OUTPUT_DIR="${SKETCHSMILES_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"
TRAIN_FRACTION="${SKETCHSMILES_TRAIN_FRACTION:-0.8}"
SEED="${SKETCHSMILES_SEED:-7}"
LIMIT="${SKETCHSMILES_LIMIT:-}"
IMAGE_SIZE="${SKETCHSMILES_IMAGE_SIZE:-256}"
SAMPLE_COUNT="${SKETCHSMILES_SAMPLE_COUNT:-64}"
CONTACT_SHEET_COLS="${SKETCHSMILES_CONTACT_SHEET_COLS:-8}"
CONTACT_THUMB_SIZE="${SKETCHSMILES_CONTACT_THUMB_SIZE:-144}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES Phase 5A-0 oracle paired-output baseline"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  run_root=$OUTPUT_DIR"
echo "  train_fraction=$TRAIN_FRACTION"
echo "  seed=$SEED"
echo "  limit=${LIMIT:-<all>}"
echo "  image_size=$IMAGE_SIZE"
echo "  sample_count=$SAMPLE_COUNT"

if [[ ! -f "$PAIR_DIR/pairs.csv" ]]; then
  echo "ERROR: pairs.csv not found under $PAIR_DIR" >&2
  echo "Run scripts/run_phase0_pairs.sh first, or set SKETCHSMILES_PAIR_DIR." >&2
  exit 2
fi

if [[ "${SKETCHSMILES_RUN_TESTS:-1}" == "1" ]]; then
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s tests
  echo
else
  echo "[1/2] Skipping tests because SKETCHSMILES_RUN_TESTS=$SKETCHSMILES_RUN_TESTS"
  echo
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import rdkit  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  echo "ERROR: Phase 5A-0 requires RDKit and Pillow for oracle rendering and consistency metrics." >&2
  echo "Try: SKETCHSMILES_MODULES=\"gcc rdkit/2025.09.4\" bash scripts/run_phase5a0_oracle_baseline.sh" >&2
  exit 2
fi

ARGS=(
  -m sketch_smiles.phase5a0_oracle_baseline
  --pair-dir "$PAIR_DIR"
  --output-dir "$OUTPUT_DIR"
  --train-fraction "$TRAIN_FRACTION"
  --seed "$SEED"
  --image-size "$IMAGE_SIZE"
  --sample-count "$SAMPLE_COUNT"
  --contact-sheet-cols "$CONTACT_SHEET_COLS"
  --contact-thumb-size "$CONTACT_THUMB_SIZE"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "[2/2] Running oracle paired-output baseline"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Phase 5A-0 oracle paired-output baseline finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/oracle_predictions.csv"
echo "  train_pairs=$OUTPUT_DIR/train_pairs.csv"
echo "  eval_pairs=$OUTPUT_DIR/eval_pairs.csv"
echo "  sample_contact_sheet=$OUTPUT_DIR/sample_contact_sheet.png"
