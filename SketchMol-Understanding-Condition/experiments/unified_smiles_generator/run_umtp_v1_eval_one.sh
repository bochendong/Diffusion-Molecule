#!/usr/bin/env bash
# Evaluate one UMTP checkpoint/task/seed using one shared maximum candidate pool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

TASK="${UMTP_EVAL_TASK:?Set UMTP_EVAL_TASK to 2p7p, ood, or table1}"
TRAIN_SEED="${UMTP_TRAIN_SEED:-7}"
EVAL_SEED="${UMTP_EVAL_SEED:-101}"
JOINT_ROOT="${UMTP_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_v1}"
CHECKPOINT="${UMTP_CHECKPOINT:-$POLICY_ROOT/seed_${TRAIN_SEED}/policy/unified_smiles_generator.pt}"
OUTPUT_ROOT="${UMTP_EVAL_OUTPUT_ROOT:-$POLICY_ROOT/eval/train_seed_${TRAIN_SEED}/eval_seed_${EVAL_SEED}/$TASK}"

case "$TASK" in
  2p7p)
    EVAL_CSV="${UMTP_EVAL_CSV:-$JOINT_ROOT/dataset/denovo_test_rows.csv}"
    FEATURES="${UMTP_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
    BENCHMARK_TASK=denovo_2p7p
    REFERENCE_CSV=""
    ;;
  ood)
    EVAL_CSV="${UMTP_EVAL_CSV:-$JOINT_ROOT/dataset/ood_test_rows.csv}"
    FEATURES="${UMTP_EVAL_FEATURES_DIR:-$JOINT_ROOT/feature_variants/ood_test_condition_features_hf_vlm}"
    BENCHMARK_TASK=denovo_ood
    REFERENCE_CSV=""
    ;;
  table1)
    EVAL_CSV="${UMTP_EVAL_CSV:-$JOINT_ROOT/dataset/table1_test_rows.csv}"
    FEATURES="${UMTP_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
    BENCHMARK_TASK=moledit_table1
    REFERENCE_CSV="$EVAL_CSV"
    ;;
  *) echo "ERROR: unsupported UMTP_EVAL_TASK=$TASK" >&2; exit 2 ;;
esac

[[ -f "$CHECKPOINT" ]] || { echo "ERROR: missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -f "$EVAL_CSV" ]] || { echo "ERROR: missing eval CSV: $EVAL_CSV" >&2; exit 2; }
[[ -d "$FEATURES" ]] || { echo "ERROR: missing feature directory: $FEATURES" >&2; exit 2; }

SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
SUCC_UNIFIED_CHECKPOINT="$CHECKPOINT" \
SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$FEATURES" \
SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$OUTPUT_ROOT" \
SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$OUTPUT_ROOT/candidate_pool" \
SUCC_UNIFIED_BENCHMARK_TASKS="$BENCHMARK_TASK" \
SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
SUCC_UNIFIED_METHOD_NAME=umtp_v1 \
SUCC_UNIFIED_DECODING_MODE=sample \
SUCC_UNIFIED_NUM_SAMPLES="${UMTP_MAX_CANDIDATES:-256}" \
SUCC_UNIFIED_MAX_CANDIDATES="${UMTP_MAX_CANDIDATES:-256}" \
SUCC_UNIFIED_TOP_K_CANDIDATES="${UMTP_MAX_CANDIDATES:-256}" \
SUCC_UNIFIED_CANDIDATE_BUDGETS="${UMTP_EVAL_BUDGETS:-1,20,128,256}" \
SUCC_UNIFIED_SELECTION_MODES="${UMTP_SELECTION_MODES:-raw,finalizer}" \
SUCC_UNIFIED_SEED="$EVAL_SEED" \
SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$REFERENCE_CSV" \
SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"

echo "UMTP v1 evaluation ready: $OUTPUT_ROOT"
