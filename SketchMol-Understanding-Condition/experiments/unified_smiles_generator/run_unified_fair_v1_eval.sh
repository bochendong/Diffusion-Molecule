#!/usr/bin/env bash
# Unified Fair Protocol v1: single-task eval with pinned CSVs and fair candidate budgets.
#
# Tasks (SUCC_UNIFIED_FAIR_TASK):
#   2p7p   - 6000 rows, direct_compat, @40, denovo_2p7p
#   ood    - 1000 rows, direct_compat, @40, denovo_ood
#   table1 - 1000 rows, direct_edit_compat, @20, moledit_table1
#
# Stages (SUCC_UNIFIED_FAIR_STAGE):
#   u0 - Direct 2p7p SFT warm-start (layout-fix inference baseline)
#   u1 - Unified de novo SFT checkpoint from fair_v1/u1_denovo_sft

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

TASK="${SUCC_UNIFIED_FAIR_TASK:?Set SUCC_UNIFIED_FAIR_TASK to 2p7p, ood, or table1}"
STAGE="${SUCC_UNIFIED_FAIR_STAGE:-u0}"
INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-with_image}"

FAIR_ROOT="${SUCC_UNIFIED_FAIR_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_fair_v1}"
SUITE_ROOT="${SUCC_UNIFIED_FAIR_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
P2P7P_ROOT="${SUCC_UNIFIED_FAIR_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
OOD_ROOT="${SUCC_UNIFIED_FAIR_OOD_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_mixed_condition}"
TABLE1_PACK="${SUCC_UNIFIED_FAIR_TABLE1_PACK:-$SUITE_ROOT/dataset/table1_eval_pack}"

DIRECT_SFT_CHECKPOINT="${SUCC_UNIFIED_FAIR_DIRECT_SFT_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"
U1_CHECKPOINT="${SUCC_UNIFIED_FAIR_U1_CHECKPOINT:-$FAIR_ROOT/u1_denovo_sft/unified_smiles_generator.pt}"

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

resolve_checkpoint() {
  local checkpoint="${SUCC_UNIFIED_CHECKPOINT:-}"
  if [[ -n "$checkpoint" ]]; then
    echo "$checkpoint"
    return
  fi
  case "$STAGE" in
  u0)
    echo "$DIRECT_SFT_CHECKPOINT"
    ;;
  u1)
    echo "$U1_CHECKPOINT"
    ;;
  *)
    echo "ERROR: unknown SUCC_UNIFIED_FAIR_STAGE=$STAGE (expected u0 or u1)" >&2
    exit 2
    ;;
  esac
}

CHECKPOINT="$(resolve_checkpoint)"
unset SUCC_UNIFIED_EVAL_LIMIT || true

case "$TASK" in
2p7p)
  BUDGET="${SUCC_UNIFIED_FAIR_BUDGET:-40}"
  OUTPUT_ROOT="${SUCC_UNIFIED_FAIR_OUTPUT_ROOT:-$FAIR_ROOT/$STAGE/2p7p/at${BUDGET}/$INPUT_MODALITY}"
  EVAL_CSV="${SUCC_UNIFIED_FAIR_2P7P_EVAL_CSV:-$P2P7P_ROOT/denovo_2p7p_eval_rows.csv}"
  EVAL_FEATURES_DIR="${SUCC_UNIFIED_FAIR_2P7P_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
  CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_compat}"
  BENCHMARK_TASKS="denovo_2p7p"
  NUM_SAMPLES="$BUDGET"
  TOP_K_CANDIDATES="$BUDGET"
  MOLEDIT_REFERENCE_CSV=""
  MOLEDIT_BUDGETS=""
  ;;
ood)
  BUDGET="${SUCC_UNIFIED_FAIR_BUDGET:-40}"
  OUTPUT_ROOT="${SUCC_UNIFIED_FAIR_OUTPUT_ROOT:-$FAIR_ROOT/$STAGE/ood/at${BUDGET}/$INPUT_MODALITY}"
  EVAL_CSV="${SUCC_UNIFIED_FAIR_OOD_EVAL_CSV:-$OOD_ROOT/denovo_ood_eval_rows.csv}"
  EVAL_FEATURES_DIR="${SUCC_UNIFIED_FAIR_OOD_EVAL_FEATURES_DIR:-$OOD_ROOT/eval_condition_features_hf_vlm}"
  CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_compat}"
  BENCHMARK_TASKS="denovo_ood"
  NUM_SAMPLES="$BUDGET"
  TOP_K_CANDIDATES="$BUDGET"
  MOLEDIT_REFERENCE_CSV=""
  MOLEDIT_BUDGETS=""
  ;;
table1)
  BUDGET="${SUCC_UNIFIED_FAIR_BUDGET:-20}"
  OUTPUT_ROOT="${SUCC_UNIFIED_FAIR_OUTPUT_ROOT:-$FAIR_ROOT/$STAGE/table1/at${BUDGET}/$INPUT_MODALITY}"
  EVAL_CSV="${SUCC_UNIFIED_FAIR_TABLE1_EVAL_CSV:-$TABLE1_PACK/table1_benchmark_condition_rows.csv}"
  MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$TABLE1_PACK/table1_moledit_rows.csv}"
  EVAL_FEATURES_DIR="${SUCC_UNIFIED_FAIR_TABLE1_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
  CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_edit_compat}"
  BENCHMARK_TASKS="moledit_table1"
  NUM_SAMPLES="$BUDGET"
  TOP_K_CANDIDATES="$BUDGET"
  MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20}"
  ;;
*)
  echo "ERROR: unknown SUCC_UNIFIED_FAIR_TASK=$TASK (expected 2p7p, ood, or table1)" >&2
  exit 2
  ;;
esac

export SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1
export SUCC_UNIFIED_CHECKPOINT="$CHECKPOINT"
export SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV"
export SUCC_UNIFIED_EVAL_FEATURES_DIR="$EVAL_FEATURES_DIR"
export SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$OUTPUT_ROOT"
export SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$OUTPUT_ROOT/sample_outputs"
export SUCC_UNIFIED_BENCHMARK_TASKS="$BENCHMARK_TASKS"
export SUCC_UNIFIED_CONDITION_LAYOUT="$CONDITION_LAYOUT"
export SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}"
export SUCC_UNIFIED_INPUT_MODALITY="$INPUT_MODALITY"
export SUCC_UNIFIED_METHOD_NAME="${SUCC_UNIFIED_METHOD_NAME:-unified_fair_v1_${STAGE}_${TASK}_at${BUDGET}_${INPUT_MODALITY}}"

export SUCC_UNIFIED_DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-sample}"
export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-$NUM_SAMPLES}"
export SUCC_UNIFIED_TEMPERATURE="${SUCC_UNIFIED_TEMPERATURE:-0.85}"
export SUCC_UNIFIED_TOP_K="${SUCC_UNIFIED_TOP_K:-40}"
export SUCC_UNIFIED_TOP_P="${SUCC_UNIFIED_TOP_P:-0.95}"
export SUCC_UNIFIED_PARALLEL_SAMPLES="${SUCC_UNIFIED_PARALLEL_SAMPLES:-8}"
export SUCC_UNIFIED_TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-$TOP_K_CANDIDATES}"

if [[ -n "$MOLEDIT_REFERENCE_CSV" ]]; then
  export SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$MOLEDIT_REFERENCE_CSV"
  export SUCC_UNIFIED_MOLEDIT_BUDGETS="$MOLEDIT_BUDGETS"
  export SUCC_UNIFIED_MOLEDIT_SELECTION_MODE="${SUCC_UNIFIED_MOLEDIT_SELECTION_MODE:-unified_score}"
  export SUCC_UNIFIED_MOLEDIT_THRESHOLDS="${SUCC_UNIFIED_MOLEDIT_THRESHOLDS:-0.65,0.15}"
  export SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-fail}"
fi

require_file "$EVAL_CSV"
require_file "$CHECKPOINT"
require_dir "$EVAL_FEATURES_DIR"
if [[ "$TASK" == "table1" ]]; then
  require_file "$MOLEDIT_REFERENCE_CSV"
fi

echo "Unified Fair v1 eval"
echo "  stage=$STAGE"
echo "  task=$TASK"
echo "  budget=@$BUDGET"
echo "  output_root=$OUTPUT_ROOT"
echo "  eval_csv=$EVAL_CSV"
echo "  eval_features_dir=$EVAL_FEATURES_DIR"
echo "  checkpoint=$CHECKPOINT"
echo "  condition_layout=$CONDITION_LAYOUT"
echo "  input_modality=$INPUT_MODALITY"
echo "  num_samples=$SUCC_UNIFIED_NUM_SAMPLES"
echo "  benchmark_tasks=$BENCHMARK_TASKS"

bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"

echo
echo "Unified Fair v1 outputs:"
echo "  suite_report=$OUTPUT_ROOT/benchmark_suite_report.md"
case "$TASK" in
2p7p) echo "  task_report=$OUTPUT_ROOT/denovo_2p7p/benchmark_report.md" ;;
ood) echo "  task_report=$OUTPUT_ROOT/denovo_ood/benchmark_report.md" ;;
table1) echo "  task_report=$OUTPUT_ROOT/moledit_table1/benchmark_report.md" ;;
esac
