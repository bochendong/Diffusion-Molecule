#!/usr/bin/env bash
# Resume MolEdit Table1 benchmark metrics for phase1 direct_edit_compat eval.
# Reuses existing sample_outputs; skips GPU sampling (RUN_SAMPLE=0).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

SUITE_ROOT="${SUCC_UNIFIED_TABLE1_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
TABLE1_PACK="${SUCC_UNIFIED_TABLE1_EVAL_PACK_DIR:-$SUITE_ROOT/dataset/table1_eval_pack}"
OUTPUT_ROOT="${SUCC_UNIFIED_TABLE1_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_phase1_table1_direct_compat_v1}"
SAMPLE_OUTPUT_DIR="${SUCC_UNIFIED_SAMPLE_OUTPUT_DIR:-$OUTPUT_ROOT/sample_outputs}"
MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$TABLE1_PACK/table1_moledit_rows.csv}"
MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20,256}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 2
  fi
}

require_file "$MOLEDIT_REFERENCE_CSV"
require_file "$SAMPLE_OUTPUT_DIR/unified_smiles_predictions.csv"
require_file "$SAMPLE_OUTPUT_DIR/unified_smiles_candidate_predictions.csv"

export SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0
export SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$OUTPUT_ROOT"
export SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$SAMPLE_OUTPUT_DIR"
export SUCC_UNIFIED_BENCHMARK_TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-moledit_table1}"
export SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$MOLEDIT_REFERENCE_CSV"
export SUCC_UNIFIED_MOLEDIT_BUDGETS="$MOLEDIT_BUDGETS"
export SUCC_UNIFIED_MOLEDIT_SELECTION_MODE="${SUCC_UNIFIED_MOLEDIT_SELECTION_MODE:-unified_score}"
export SUCC_UNIFIED_MOLEDIT_THRESHOLDS="${SUCC_UNIFIED_MOLEDIT_THRESHOLDS:-0.65,0.15}"
export SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-fail}"
export SUCC_UNIFIED_MOLEDIT_RDKIT_MODULES="${SUCC_UNIFIED_MOLEDIT_RDKIT_MODULES:-gcc/12.3 rdkit/2024.09.6}"
export SUCC_UNIFIED_METHOD_NAME="${SUCC_UNIFIED_METHOD_NAME:-unified_phase1_table1_direct_compat}"

echo "Unified phase1 Table1 benchmark resume (metrics only)"
echo "  output_root=$OUTPUT_ROOT"
echo "  sample_output_dir=$SAMPLE_OUTPUT_DIR"
echo "  moledit_reference_csv=$MOLEDIT_REFERENCE_CSV"
echo "  moledit_budgets=$MOLEDIT_BUDGETS"
echo "  moledit_rdkit_modules=$SUCC_UNIFIED_MOLEDIT_RDKIT_MODULES"

bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"

echo
echo "Unified phase1 Table1 benchmark resume outputs:"
echo "  report=$OUTPUT_ROOT/moledit_table1/benchmark_report.md"
echo "  suite_report=$OUTPUT_ROOT/benchmark_suite_report.md"
