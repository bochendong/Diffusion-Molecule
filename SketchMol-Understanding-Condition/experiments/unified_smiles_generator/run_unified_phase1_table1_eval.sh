#!/usr/bin/env bash
# Phase-1 fixed direct_compat eval on MolEdit Table1 rows (1000 eval rows).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SUITE_ROOT="${SUCC_UNIFIED_TABLE1_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
TABLE1_PACK="${SUCC_UNIFIED_TABLE1_EVAL_PACK_DIR:-$SUITE_ROOT/dataset/table1_eval_pack}"
OUTPUT_ROOT="${SUCC_UNIFIED_TABLE1_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_phase1_table1_direct_compat_v1}"
EVAL_CSV="${SUCC_UNIFIED_TABLE1_EVAL_CSV:-$TABLE1_PACK/table1_benchmark_condition_rows.csv}"
MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$TABLE1_PACK/table1_moledit_rows.csv}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_TABLE1_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"

GROUP_RL_CHECKPOINT="${SUCC_UNIFIED_TABLE1_GROUP_RL_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_moledit_table1_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
SFT_CHECKPOINT="${SUCC_UNIFIED_TABLE1_SFT_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/direct_smiles_model/direct_smiles_generator.pt}"
CHECKPOINT="${SUCC_UNIFIED_CHECKPOINT:-}"
if [[ -z "$CHECKPOINT" ]]; then
  if [[ -f "$GROUP_RL_CHECKPOINT" ]]; then
    CHECKPOINT="$GROUP_RL_CHECKPOINT"
  elif [[ -f "$SFT_CHECKPOINT" ]]; then
    CHECKPOINT="$SFT_CHECKPOINT"
    echo "WARNING: moledit group RL checkpoint missing; falling back to de novo SFT: $SFT_CHECKPOINT" >&2
  else
    echo "ERROR: no checkpoint found (tried group RL and de novo SFT)" >&2
    exit 2
  fi
fi

CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_edit_compat}"
CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}"
INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-with_image}"
MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20,256}"

export SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1
export SUCC_UNIFIED_CHECKPOINT="$CHECKPOINT"
export SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV"
export SUCC_UNIFIED_EVAL_FEATURES_DIR="$EVAL_FEATURES_DIR"
export SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$OUTPUT_ROOT"
export SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$OUTPUT_ROOT/sample_outputs"
export SUCC_UNIFIED_BENCHMARK_TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-moledit_table1}"
export SUCC_UNIFIED_CONDITION_LAYOUT="$CONDITION_LAYOUT"
export SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="$CONDITION_FEATURE_VARIANT"
export SUCC_UNIFIED_INPUT_MODALITY="$INPUT_MODALITY"
export SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$MOLEDIT_REFERENCE_CSV"
export SUCC_UNIFIED_MOLEDIT_BUDGETS="$MOLEDIT_BUDGETS"
export SUCC_UNIFIED_MOLEDIT_SELECTION_MODE="${SUCC_UNIFIED_MOLEDIT_SELECTION_MODE:-unified_score}"
export SUCC_UNIFIED_MOLEDIT_THRESHOLDS="${SUCC_UNIFIED_MOLEDIT_THRESHOLDS:-0.65,0.15}"
export SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-fail}"
export SUCC_UNIFIED_METHOD_NAME="${SUCC_UNIFIED_METHOD_NAME:-unified_phase1_table1_direct_compat}"

export SUCC_UNIFIED_DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-sample}"
export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-256}"
export SUCC_UNIFIED_TEMPERATURE="${SUCC_UNIFIED_TEMPERATURE:-0.85}"
export SUCC_UNIFIED_TOP_K="${SUCC_UNIFIED_TOP_K:-40}"
export SUCC_UNIFIED_TOP_P="${SUCC_UNIFIED_TOP_P:-0.95}"
export SUCC_UNIFIED_PARALLEL_SAMPLES="${SUCC_UNIFIED_PARALLEL_SAMPLES:-8}"
export SUCC_UNIFIED_TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-256}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: missing required directory: $1" >&2
    exit 2
  fi
}

require_file "$EVAL_CSV"
require_file "$MOLEDIT_REFERENCE_CSV"
require_file "$CHECKPOINT"
require_dir "$EVAL_FEATURES_DIR"

echo "Unified phase1 Table1 direct_edit_compat eval"
echo "  output_root=$OUTPUT_ROOT"
echo "  eval_csv=$EVAL_CSV"
echo "  moledit_reference_csv=$MOLEDIT_REFERENCE_CSV"
echo "  eval_features_dir=$EVAL_FEATURES_DIR"
echo "  checkpoint=$CHECKPOINT"
echo "  condition_layout=$CONDITION_LAYOUT"
echo "  input_modality=$INPUT_MODALITY"
echo "  decoding_mode=$SUCC_UNIFIED_DECODING_MODE"
echo "  num_samples=$SUCC_UNIFIED_NUM_SAMPLES"
echo "  moledit_budgets=$MOLEDIT_BUDGETS"
echo "  eval_limit=${SUCC_UNIFIED_EVAL_LIMIT:-all}"

bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"

echo
echo "Unified phase1 Table1 outputs:"
echo "  report=$OUTPUT_ROOT/moledit_table1/benchmark_report.md"
echo "  suite_report=$OUTPUT_ROOT/benchmark_suite_report.md"
