#!/usr/bin/env bash
# Join SketchMol shard OCR outputs and evaluate the de novo 2p-7p benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# The benchmark evaluator canonicalizes and scores SMILES with RDKit. Compute
# jobs do not inherit the interactive module environment, so initialize the
# same cluster RDKit module used by the shard runner.
SKETCHMOL_BENCHMARK_MODULES="${SKETCHMOL_BENCHMARK_MODULES:-gcc rdkit/2024.09.6}"
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck source=/dev/null
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
fi
if command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_BENCHMARK_MODULES
fi

PYTHON_BIN="${SKETCHMOL_BENCHMARK_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
EVAL_CSV="${SKETCHMOL_DENOVO_EVAL_CSV:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv}"
OUTPUT_DIR="${SKETCHMOL_DENOVO_OUTPUT_DIR:-SketchMolBenchmark/outputs/sketchmol_denovo_2p7p_at40_v1}"
CANDIDATE_BUDGET="${SKETCHMOL_DENOVO_CONDITIONAL_COUNT:-40}"
JOINED_CSV="${SKETCHMOL_DENOVO_JOINED_CSV:-$OUTPUT_DIR/sketchmol_denovo_2p7p_joined.csv}"
BENCHMARK_OUTPUT_DIR="${SKETCHMOL_DENOVO_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_eval}"
SHARDS_DIR="$OUTPUT_DIR/shards"

export PYTHONPATH="$REPO_ROOT/SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"

echo "Benchmark modules: $SKETCHMOL_BENCHMARK_MODULES"
echo "Joining SketchMol shard candidates..."
"$PYTHON_BIN" "$SCRIPT_DIR/join_denovo_2p7p_sketchmol_candidates.py" \
  --eval-csv "$EVAL_CSV" \
  --shards-dir "$SHARDS_DIR" \
  --output-csv "$JOINED_CSV" \
  --candidate-budget "$CANDIDATE_BUDGET" \
  --summary-json "$OUTPUT_DIR/sketchmol_denovo_2p7p_join.summary.json"

echo "Evaluating joined SketchMol OCR predictions..."
"$PYTHON_BIN" "$REPO_ROOT/SketchMol-Understanding-Condition/scripts/evaluate_univideo_image_benchmark.py" \
  --image-csv "$JOINED_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --method "sketchmol_real_ocr_n${CANDIDATE_BUDGET}" \
  --smiles-column generated_smiles \
  --report-title "SketchMol Real OCR De Novo 2p-7p Benchmark" \
  --benchmark-family "sketchmol_real_ocr" \
  --benchmark-task "sketchmol_denovo_2p7p_property_design" \
  --accept-direct-smiles \
  --hide-source-similarity-section

echo
echo "SketchMol de novo 2p-7p eval ready:"
echo "  joined_csv=$JOINED_CSV"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
sed -n '1,100p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
