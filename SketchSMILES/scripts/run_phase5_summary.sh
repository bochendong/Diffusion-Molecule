#!/usr/bin/env bash
# Summarize SketchSMILES Phase 5 run metrics into one table.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUTPUT_DIR="${SKETCHSMILES_SUMMARY_DIR:-outputs/summary/phase5}"
DEFAULT_RUN_DIRS=(
  outputs/runs/phase5a1_learned_smiles_decoder_seed7
  outputs/runs/phase5a2_tokenized_beam_decoder_seed7
  outputs/runs/phase5a4_reranked_transformer_decoder_seed7
  outputs/runs/phase5a6_randomized_smiles_decoder_seed7
)

if [[ -n "${SKETCHSMILES_RUN_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_DIRS=($SKETCHSMILES_RUN_DIRS)
else
  RUN_DIRS=("${DEFAULT_RUN_DIRS[@]}")
fi

echo "SketchSMILES Phase 5 summary"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  run_dirs=${RUN_DIRS[*]}"

"$PYTHON_BIN" -m sketch_smiles.phase5_summary --output-dir "$OUTPUT_DIR" "${RUN_DIRS[@]}"

echo
echo "Phase 5 summary finished: $OUTPUT_DIR"
echo "  csv=$OUTPUT_DIR/phase5_summary.csv"
echo "  json=$OUTPUT_DIR/phase5_summary.json"
