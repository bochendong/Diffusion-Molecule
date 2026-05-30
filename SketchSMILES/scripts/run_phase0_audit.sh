#!/usr/bin/env bash
# Audit Phase 0 paired SMILES/rendered-image manifest.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
SAMPLE_COUNT="${SKETCHSMILES_SAMPLE_COUNT:-64}"
EXPECTED_IMAGE_SIZE="${SKETCHSMILES_EXPECTED_IMAGE_SIZE:-256}"
CONTACT_SHEET_COLS="${SKETCHSMILES_CONTACT_SHEET_COLS:-8}"
CONTACT_THUMB_SIZE="${SKETCHSMILES_CONTACT_THUMB_SIZE:-160}"
SEED="${SKETCHSMILES_SEED:-7}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES Phase 0 paired manifest audit"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  sample_count=$SAMPLE_COUNT"
echo "  expected_image_size=$EXPECTED_IMAGE_SIZE"
echo "  seed=$SEED"

if [[ ! -f "$PAIR_DIR/pairs.csv" ]]; then
  echo "ERROR: pairs.csv not found under $PAIR_DIR" >&2
  echo "Run scripts/run_phase0_pairs.sh first, or set SKETCHSMILES_PAIR_DIR." >&2
  exit 2
fi

"$PYTHON_BIN" -m sketch_smiles.audit_pairs \
  --pair-dir "$PAIR_DIR" \
  --sample-count "$SAMPLE_COUNT" \
  --seed "$SEED" \
  --expected-image-size "$EXPECTED_IMAGE_SIZE" \
  --contact-sheet-cols "$CONTACT_SHEET_COLS" \
  --contact-thumb-size "$CONTACT_THUMB_SIZE"

echo
echo "Phase 0 audit finished: $PAIR_DIR"
echo "  audit_summary=$PAIR_DIR/audit_summary.json"
echo "  audit_rows=$PAIR_DIR/audit_rows.csv"
echo "  sample_pairs=$PAIR_DIR/sample_pairs.csv"
echo "  sample_contact_sheet=$PAIR_DIR/sample_contact_sheet.png"
