#!/usr/bin/env bash
# Run Phase 5A-5 retrieval diagnostic on an existing 5A-4 run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_DIR="${SKETCHSMILES_SOURCE_RUN_DIR:-outputs/runs/phase5a4_reranked_transformer_decoder_seed7}"
OUTPUT_DIR="${SKETCHSMILES_OUTPUT_DIR:-$RUN_DIR/retrieval_diagnostic}"
FINGERPRINT_BITS="${SKETCHSMILES_FINGERPRINT_BITS:-2048}"
RETRIEVAL_TOP_K="${SKETCHSMILES_RETRIEVAL_TOP_K:-16}"
MAX_EVAL="${SKETCHSMILES_MAX_EVAL:-}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES Phase 5A-5 retrieval diagnostic"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"
echo "  source_run_dir=$RUN_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  fingerprint_bits=$FINGERPRINT_BITS"
echo "  retrieval_top_k=$RETRIEVAL_TOP_K"
echo "  max_eval=${MAX_EVAL:-<all>}"

if [[ ! -f "$RUN_DIR/predictions.csv" ]]; then
  echo "ERROR: predictions.csv not found in $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/train_pairs.csv" ]]; then
  echo "ERROR: train_pairs.csv not found in $RUN_DIR" >&2
  exit 2
fi

ARGS=(
  -m sketch_smiles.phase5a5_retrieval_diagnostic
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --fingerprint-bits "$FINGERPRINT_BITS"
  --retrieval-top-k "$RETRIEVAL_TOP_K"
)
if [[ -n "$MAX_EVAL" ]]; then
  ARGS+=(--max-eval "$MAX_EVAL")
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Phase 5A-5 retrieval diagnostic finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  summary=$OUTPUT_DIR/source_summary.csv"
echo "  detail=$OUTPUT_DIR/retrieval_diagnostic_rows.csv"
