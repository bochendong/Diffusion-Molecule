#!/usr/bin/env bash
# Build Phase 0 paired SMILES/rendered-image manifest.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
INPUT_CSV="${SKETCHSMILES_INPUT_CSV:-data/example_molecules.csv}"
OUTPUT_DIR="${SKETCHSMILES_OUTPUT_DIR:-outputs/pairs/phase0_smoke}"
SMILES_COLUMN="${SKETCHSMILES_SMILES_COLUMN:-smiles}"
IMAGE_SIZE="${SKETCHSMILES_IMAGE_SIZE:-256}"
LIMIT="${SKETCHSMILES_LIMIT:-}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES Phase 0 paired manifest"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"
echo "  input_csv=$INPUT_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  smiles_column=$SMILES_COLUMN"
echo "  image_size=$IMAGE_SIZE"
echo "  limit=${LIMIT:-<all>}"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import rdkit  # noqa: F401
PY
then
  echo "ERROR: RDKit unavailable for $PYTHON_BIN; Phase 0 needs RDKit rendering." >&2
  echo "Try: SKETCHSMILES_MODULES=\"gcc rdkit/2025.09.4\" bash scripts/run_phase0_pairs.sh" >&2
  exit 2
fi

ARGS=(
  -m sketch_smiles.build_pairs
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --smiles-column "$SMILES_COLUMN"
  --image-size "$IMAGE_SIZE"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Phase 0 paired manifest finished: $OUTPUT_DIR"
echo "  pairs=$OUTPUT_DIR/pairs.csv"
echo "  summary=$OUTPUT_DIR/summary.json"
echo "  images=$OUTPUT_DIR/images"
