#!/usr/bin/env bash
# Formal Unified Joint v2 evaluation: one 256-candidate pool per run, offline prefixes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

JOINT_ROOT="${SUCC_UNIFIED_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${SUCC_UNIFIED_JOINT_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
P2P7P_ROOT="${SUCC_UNIFIED_JOINT_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
DATA_ROOT="${SUCC_UNIFIED_JOINT_DATA_ROOT:-$JOINT_ROOT/dataset}"
EVAL_ROOT="${SUCC_UNIFIED_JOINT_EVAL_ROOT:-$JOINT_ROOT/eval}"

STAGES_RAW="${SUCC_UNIFIED_JOINT_EVAL_STAGES:-u0,u1,u2}"
TRAIN_SEEDS_RAW="${SUCC_UNIFIED_TRAIN_SEEDS:-7,17,27}"
EVAL_SEEDS_RAW="${SUCC_UNIFIED_EVAL_SEEDS:-101,202,303}"
TASKS_RAW="${SUCC_UNIFIED_JOINT_EVAL_TASKS:-2p7p,ood,table1}"
BUDGETS="${SUCC_UNIFIED_JOINT_EVAL_BUDGETS:-1,20,128,256}"
SELECTIONS="${SUCC_UNIFIED_JOINT_SELECTION_MODES:-raw,finalizer}"
MAX_CANDIDATES="${SUCC_UNIFIED_MAX_CANDIDATES:-256}"

DENOVO_TEST_CSV="${SUCC_UNIFIED_JOINT_DENOVO_TEST_CSV:-$DATA_ROOT/denovo_test_rows.csv}"
TABLE1_TEST_CSV="${SUCC_UNIFIED_JOINT_TABLE1_TEST_CSV:-$DATA_ROOT/table1_test_rows.csv}"
OOD_TEST_CSV="${SUCC_UNIFIED_JOINT_OOD_TEST_CSV:-$DATA_ROOT/ood_test_rows.csv}"
ID_FEATURES_DIR="${SUCC_UNIFIED_JOINT_ID_TEST_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
OOD_FEATURES_DIR="${SUCC_UNIFIED_JOINT_OOD_TEST_FEATURES_DIR:-$JOINT_ROOT/feature_variants/ood_test_condition_features_hf_vlm}"
BASE_CHECKPOINT="${SUCC_UNIFIED_JOINT_BASE_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"

require_file() { [[ -f "$1" ]] || { echo "ERROR: missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "ERROR: missing directory: $1" >&2; exit 2; }; }

require_file "$DENOVO_TEST_CSV"
require_file "$TABLE1_TEST_CSV"
require_file "$OOD_TEST_CSV"
require_file "$BASE_CHECKPOINT"
require_dir "$ID_FEATURES_DIR"
require_dir "$OOD_FEATURES_DIR"

IFS=',' read -r -a STAGES <<< "$STAGES_RAW"
IFS=',' read -r -a TRAIN_SEEDS <<< "$TRAIN_SEEDS_RAW"
IFS=',' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_RAW"
IFS=',' read -r -a TASKS <<< "$TASKS_RAW"

for stage in "${STAGES[@]}"; do
  if [[ "$stage" == "u0" ]]; then
    stage_train_seeds=(base)
  else
    stage_train_seeds=("${TRAIN_SEEDS[@]}")
  fi
  for train_seed in "${stage_train_seeds[@]}"; do
    case "$stage" in
    u0) checkpoint="$BASE_CHECKPOINT" ;;
    u1) checkpoint="$JOINT_ROOT/u1_joint_sft/seed_${train_seed}/selected_checkpoint.pt" ;;
    u2) checkpoint="$JOINT_ROOT/u2_joint_protected_sft/seed_${train_seed}/selected_checkpoint.pt" ;;
    *) echo "ERROR: stage must be u0, u1, or u2: $stage" >&2; exit 2 ;;
    esac
    require_file "$checkpoint"
    for eval_seed in "${EVAL_SEEDS[@]}"; do
      for task in "${TASKS[@]}"; do
        case "$task" in
        2p7p)
          eval_csv="$DENOVO_TEST_CSV"
          features_dir="$ID_FEATURES_DIR"
          reference_csv=""
          ;;
        ood)
          eval_csv="$OOD_TEST_CSV"
          features_dir="$OOD_FEATURES_DIR"
          reference_csv=""
          ;;
        table1)
          eval_csv="$TABLE1_TEST_CSV"
          features_dir="$ID_FEATURES_DIR"
          reference_csv="$TABLE1_TEST_CSV"
          ;;
        *) echo "ERROR: unsupported task=$task" >&2; exit 2 ;;
        esac
        output_root="$EVAL_ROOT/$stage/train_seed_${train_seed}/eval_seed_${eval_seed}/$task"
        echo "=== stage=$stage train_seed=$train_seed eval_seed=$eval_seed task=$task ==="
        SUCC_UNIFIED_JOINT_EVAL_TASK="$task" \
        SUCC_UNIFIED_JOINT_STAGE="$stage" \
        SUCC_UNIFIED_TRAIN_SEED="$train_seed" \
        SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
        SUCC_UNIFIED_EVAL_CSV="$eval_csv" \
        SUCC_UNIFIED_EVAL_FEATURES_DIR="$features_dir" \
        SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_root" \
        SUCC_UNIFIED_CANDIDATE_BUDGETS="$BUDGETS" \
        SUCC_UNIFIED_MAX_CANDIDATES="$MAX_CANDIDATES" \
        SUCC_UNIFIED_SELECTION_MODES="$SELECTIONS" \
        SUCC_UNIFIED_SEED="$eval_seed" \
        SUCC_UNIFIED_METHOD_NAME="unified_joint_v2_${stage}_${task}" \
        SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$reference_csv" \
        SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
        bash "$SCRIPT_DIR/run_unified_joint_v2_eval_one.sh"
      done
    done
  done
done

if [[ "${SUCC_UNIFIED_JOINT_SKIP_COLLECT:-0}" != "1" ]]; then
  "${SUCC_PYTHON_BIN:-python3}" "$SCRIPT_DIR/collect_unified_joint_v2_results.py" \
    --eval-root "$EVAL_ROOT" \
    --selection-root "$JOINT_ROOT" \
    --output-prefix "$EVAL_ROOT/unified_joint_v2"
fi

echo "Unified Joint v2 formal evaluation complete: $EVAL_ROOT"
