#!/usr/bin/env bash
# Train one true joint Unified checkpoint on balanced de novo 2p-7p + Table1 edit rows.
#
# Stages:
#   u1 - joint task-balanced SFT
#   u2 - joint task-balanced SFT with frozen de novo teacher KL protection

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
STAGE="${SUCC_UNIFIED_JOINT_STAGE:-u2}"
TRAIN_SEED="${SUCC_UNIFIED_SEED:-7}"
JOINT_ROOT="${SUCC_UNIFIED_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${SUCC_UNIFIED_JOINT_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
P2P7P_ROOT="${SUCC_UNIFIED_JOINT_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
DATA_ROOT="${SUCC_UNIFIED_JOINT_DATA_ROOT:-$JOINT_ROOT/dataset}"
MOLECULE_DB="${SUCC_UNIFIED_JOINT_MOLECULE_DB:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/molecule_database.csv}"

SOURCE_TRAIN_CSV="${SUCC_UNIFIED_JOINT_SOURCE_TRAIN_CSV:-$SUITE_ROOT/dataset/unified_train_rows.csv}"
SOURCE_EVAL_CSV="${SUCC_UNIFIED_JOINT_SOURCE_EVAL_CSV:-$SUITE_ROOT/dataset/unified_eval_rows.csv}"
SOURCE_CANDIDATE_CSV="${SUCC_UNIFIED_JOINT_SOURCE_CANDIDATE_CSV:-$P2P7P_ROOT/denovo_2p7p_candidate_rows.csv}"
TRAIN_CSV="${SUCC_UNIFIED_JOINT_TRAIN_CSV:-$DATA_ROOT/unified_joint_train_rows.csv}"
VALIDATION_CSV="${SUCC_UNIFIED_JOINT_VALIDATION_CSV:-$DATA_ROOT/unified_joint_validation_rows.csv}"
DENOVO_VALIDATION_CSV="${SUCC_UNIFIED_JOINT_DENOVO_VALIDATION_CSV:-$DATA_ROOT/denovo_validation_rows.csv}"
TABLE1_VALIDATION_CSV="${SUCC_UNIFIED_JOINT_TABLE1_VALIDATION_CSV:-$DATA_ROOT/table1_validation_rows.csv}"
DENOVO_TEST_CSV="${SUCC_UNIFIED_JOINT_DENOVO_TEST_CSV:-$DATA_ROOT/denovo_test_rows.csv}"
TABLE1_TEST_CSV="${SUCC_UNIFIED_JOINT_TABLE1_TEST_CSV:-$DATA_ROOT/table1_test_rows.csv}"
OOD_FAIR_ROOT="${SUCC_UNIFIED_JOINT_OOD_FAIR_ROOT:-$DATA_ROOT/ood_molecule_disjoint}"
OOD_SOURCE_CSV="${SUCC_UNIFIED_JOINT_OOD_SOURCE_CSV:-$OOD_FAIR_ROOT/denovo_ood_eval_rows_raw.csv}"
OOD_TEST_CSV="${SUCC_UNIFIED_JOINT_OOD_TEST_CSV:-$DATA_ROOT/ood_test_rows.csv}"
MANIFEST_JSON="${SUCC_UNIFIED_JOINT_MANIFEST_JSON:-$DATA_ROOT/unified_joint_manifest.json}"
TRAIN_FEATURES_DIR="${SUCC_UNIFIED_JOINT_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
VALIDATION_FEATURES_DIR="${SUCC_UNIFIED_JOINT_VALIDATION_FEATURES_DIR:-$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm}"
OOD_TEST_FEATURES_DIR="${SUCC_UNIFIED_JOINT_OOD_TEST_FEATURES_DIR:-$JOINT_ROOT/feature_variants/ood_test_condition_features_hf_vlm}"
BASE_CHECKPOINT="${SUCC_UNIFIED_JOINT_BASE_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"

case "$STAGE" in
u1)
  OUTPUT_DIR="${SUCC_UNIFIED_JOINT_OUTPUT_DIR:-$JOINT_ROOT/u1_joint_sft/seed_${TRAIN_SEED}}"
  TEACHER_CHECKPOINT=""
  DISTILL_WEIGHT="${SUCC_UNIFIED_DISTILL_WEIGHT:-0}"
  ;;
u2)
  OUTPUT_DIR="${SUCC_UNIFIED_JOINT_OUTPUT_DIR:-$JOINT_ROOT/u2_joint_protected_sft/seed_${TRAIN_SEED}}"
  TEACHER_CHECKPOINT="${SUCC_UNIFIED_TEACHER_CHECKPOINT:-$BASE_CHECKPOINT}"
  DISTILL_WEIGHT="${SUCC_UNIFIED_DISTILL_WEIGHT:-0.3}"
  ;;
*)
  echo "ERROR: unknown SUCC_UNIFIED_JOINT_STAGE=$STAGE (expected u1 or u2)" >&2
  exit 2
  ;;
esac

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

require_file "$SOURCE_TRAIN_CSV"
require_file "$SOURCE_EVAL_CSV"
require_file "$SOURCE_CANDIDATE_CSV"
require_file "$MOLECULE_DB"
require_file "$BASE_CHECKPOINT"
require_dir "$TRAIN_FEATURES_DIR"
if [[ -n "$TEACHER_CHECKPOINT" ]]; then
  require_file "$TEACHER_CHECKPOINT"
fi

mkdir -p "$DATA_ROOT" "$OUTPUT_DIR" "$OOD_FAIR_ROOT"

if [[ "${SUCC_UNIFIED_JOINT_SKIP_PREPARE:-0}" != "1" && ( "${SUCC_UNIFIED_JOINT_REBUILD_OOD:-0}" == "1" || ! -f "$OOD_SOURCE_CSV" ) ]]; then
  echo "=== rebuild molecule-disjoint OOD test rows ==="
  "$PYTHON_BIN" SketchMol-Understanding-Condition/scripts/export_denovo_ood_benchmark_rows.py \
    --molecule-db-csv "$MOLECULE_DB" \
    --output-csv "$OOD_SOURCE_CSV" \
    --negative-output-csv "$OOD_FAIR_ROOT/denovo_ood_negative_rows.csv" \
    --candidate-output-csv "$OOD_FAIR_ROOT/denovo_ood_candidate_rows.csv" \
    --rows-per-spec 100 \
    --exclude-molecule-csv "$SOURCE_TRAIN_CSV" \
    --seed 37
fi
require_file "$OOD_SOURCE_CSV"

if [[ "${SUCC_UNIFIED_JOINT_SKIP_PREPARE:-0}" != "1" ]]; then
  echo "=== prepare leakage-audited joint rows ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_unified_joint_rows.py" \
    --source-train-csv "$SOURCE_TRAIN_CSV" \
    --source-eval-csv "$SOURCE_EVAL_CSV" \
    --source-candidate-csv "$SOURCE_CANDIDATE_CSV" \
    --ood-eval-csv "$OOD_SOURCE_CSV" \
    --train-output-csv "$TRAIN_CSV" \
    --validation-output-csv "$VALIDATION_CSV" \
    --denovo-validation-output-csv "$DENOVO_VALIDATION_CSV" \
    --table1-validation-output-csv "$TABLE1_VALIDATION_CSV" \
    --denovo-test-output-csv "$DENOVO_TEST_CSV" \
    --table1-test-output-csv "$TABLE1_TEST_CSV" \
    --ood-test-output-csv "$OOD_TEST_CSV" \
    --manifest-json "$MANIFEST_JSON" \
    --denovo-train-per-count "${SUCC_UNIFIED_JOINT_DENOVO_TRAIN_PER_COUNT:-2000}" \
    --denovo-validation-per-count "${SUCC_UNIFIED_JOINT_DENOVO_VALIDATION_PER_COUNT:-200}" \
    --denovo-test-per-count "${SUCC_UNIFIED_JOINT_DENOVO_TEST_PER_COUNT:-1000}" \
    --edit-train-per-task "${SUCC_UNIFIED_JOINT_EDIT_TRAIN_PER_TASK:-450}" \
    --edit-validation-per-task "${SUCC_UNIFIED_JOINT_EDIT_VALIDATION_PER_TASK:-50}" \
    --edit-test-per-task "${SUCC_UNIFIED_JOINT_EDIT_TEST_PER_TASK:-100}" \
    --ood-test-per-spec "${SUCC_UNIFIED_JOINT_OOD_TEST_PER_SPEC:-100}" \
    --overlap-policy "${SUCC_UNIFIED_JOINT_OVERLAP_POLICY:-drop_train}" \
    --seed "${SUCC_UNIFIED_JOINT_DATA_SEED:-41}"
fi

export_features() {
  local rows_csv="$1"
  local features_dir="$2"
  if [[ -f "$features_dir/query_tokens.npy" && "${SUCC_UNIFIED_JOINT_REBUILD_FEATURES:-0}" != "1" ]]; then
    return
  fi
  SUCC_ENCODER=hf_vlm \
  SUCC_VARIANTS=full \
  SUCC_BASELINE_CSV="$rows_csv" \
  SUCC_OUTPUT_DIR="$features_dir" \
  bash SketchMol-Understanding-Condition/scripts/run_condition_encoder_export.sh
}

echo
echo "=== export validation/OOD frozen condition features ==="
if [[ "${SUCC_UNIFIED_JOINT_SKIP_PREPARE:-0}" != "1" ]]; then
  export_features "$VALIDATION_CSV" "$VALIDATION_FEATURES_DIR"
  export_features "$OOD_TEST_CSV" "$OOD_TEST_FEATURES_DIR"
fi

require_file "$TRAIN_CSV"
require_file "$VALIDATION_CSV"
require_file "$DENOVO_VALIDATION_CSV"
require_file "$TABLE1_VALIDATION_CSV"
require_file "$DENOVO_TEST_CSV"
require_file "$TABLE1_TEST_CSV"
require_file "$MANIFEST_JSON"
require_dir "$VALIDATION_FEATURES_DIR"
require_dir "$OOD_TEST_FEATURES_DIR"

if [[ "${SUCC_UNIFIED_JOINT_PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Unified Joint v2 data preparation complete: $MANIFEST_JSON"
  exit 0
fi

echo
echo "=== Unified Joint v2 training: stage=$STAGE ==="
SUCC_UNIFIED_TRAIN_CSV="$TRAIN_CSV" \
SUCC_UNIFIED_EVAL_CSV="$VALIDATION_CSV" \
SUCC_UNIFIED_OUTPUT_DIR="$OUTPUT_DIR" \
SUCC_UNIFIED_TRAIN_FEATURES_DIR="$TRAIN_FEATURES_DIR" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$VALIDATION_FEATURES_DIR" \
SUCC_UNIFIED_CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_compat}" \
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}" \
SUCC_UNIFIED_INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-mixed_by_task}" \
SUCC_UNIFIED_RESUME_CHECKPOINT="$BASE_CHECKPOINT" \
SUCC_UNIFIED_RESET_TRAINING_STATE=1 \
SUCC_UNIFIED_SAMPLING_MODE=task_balanced \
SUCC_UNIFIED_SAMPLES_PER_EPOCH="${SUCC_UNIFIED_SAMPLES_PER_EPOCH:-24000}" \
SUCC_UNIFIED_TEACHER_CHECKPOINT="$TEACHER_CHECKPOINT" \
SUCC_UNIFIED_DISTILL_WEIGHT="$DISTILL_WEIGHT" \
SUCC_UNIFIED_DISTILL_TEMPERATURE="${SUCC_UNIFIED_DISTILL_TEMPERATURE:-1.0}" \
SUCC_UNIFIED_EPOCHS="${SUCC_UNIFIED_EPOCHS:-4}" \
SUCC_UNIFIED_BATCH_SIZE="${SUCC_UNIFIED_BATCH_SIZE:-64}" \
SUCC_UNIFIED_EVAL_BATCH_SIZE="${SUCC_UNIFIED_EVAL_BATCH_SIZE:-128}" \
SUCC_UNIFIED_LR="${SUCC_UNIFIED_LR:-1e-4}" \
SUCC_UNIFIED_EVAL_LIMIT="${SUCC_UNIFIED_EVAL_LIMIT:-0}" \
SUCC_UNIFIED_SEED="$TRAIN_SEED" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_train.sh"

if [[ "${SUCC_UNIFIED_JOINT_RUN_VALIDATION_GATE:-1}" == "1" ]]; then
  echo
  echo "=== Unified Joint v2 validation-only checkpoint selection ==="
  BASELINE_SUMMARY="$JOINT_ROOT/validation/u0/base/eval_seed_101/2p7p/denovo_2p7p/n20/finalizer/benchmark_summary.csv"
  if [[ ! -f "$BASELINE_SUMMARY" ]]; then
    SUCC_UNIFIED_JOINT_STAGE=u0 \
    SUCC_UNIFIED_VALIDATION_SEEDS="${SUCC_UNIFIED_VALIDATION_SEEDS:-101,202,303}" \
    bash "$SCRIPT_DIR/run_unified_joint_v2_validation.sh"
  fi
  SUCC_UNIFIED_JOINT_STAGE="$STAGE" \
  SUCC_UNIFIED_TRAIN_SEED="$TRAIN_SEED" \
  SUCC_UNIFIED_VALIDATION_SEEDS="${SUCC_UNIFIED_VALIDATION_SEEDS:-101,202,303}" \
  bash "$SCRIPT_DIR/run_unified_joint_v2_validation.sh"

  "$PYTHON_BIN" "$SCRIPT_DIR/select_unified_joint_checkpoint.py" \
    --baseline-root "$JOINT_ROOT/validation/u0/base" \
    --candidate-root "$JOINT_ROOT/validation/$STAGE/seed_${TRAIN_SEED}" \
    --checkpoint-dir "$OUTPUT_DIR" \
    --output-json "$OUTPUT_DIR/checkpoint_selection.json" \
    --selected-checkpoint "$OUTPUT_DIR/selected_checkpoint.pt" \
    --max-denovo-drop "${SUCC_UNIFIED_MAX_DENOVO_DROP:-0.02}"
fi

echo
echo "Unified Joint v2 ready:"
echo "  stage=$STAGE"
echo "  train_seed=$TRAIN_SEED"
echo "  latest_checkpoint=$OUTPUT_DIR/unified_smiles_generator.pt"
echo "  selected_checkpoint=$OUTPUT_DIR/selected_checkpoint.pt"
echo "  checkpoint_selection=$OUTPUT_DIR/checkpoint_selection.json"
echo "  manifest=$MANIFEST_JSON"
echo "  train_csv=$TRAIN_CSV"
echo "  validation_csv=$VALIDATION_CSV"
echo "  denovo_test_csv=$DENOVO_TEST_CSV"
echo "  table1_test_csv=$TABLE1_TEST_CSV"
echo "  ood_test_csv=$OOD_TEST_CSV"
echo "  distill_weight=$DISTILL_WEIGHT"
