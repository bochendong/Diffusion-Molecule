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
JOINT_ROOT="${SUCC_UNIFIED_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${SUCC_UNIFIED_JOINT_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
P2P7P_ROOT="${SUCC_UNIFIED_JOINT_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
DATA_ROOT="${SUCC_UNIFIED_JOINT_DATA_ROOT:-$JOINT_ROOT/dataset}"

SOURCE_TRAIN_CSV="${SUCC_UNIFIED_JOINT_SOURCE_TRAIN_CSV:-$SUITE_ROOT/dataset/unified_train_rows.csv}"
SOURCE_EVAL_CSV="${SUCC_UNIFIED_JOINT_SOURCE_EVAL_CSV:-$SUITE_ROOT/dataset/unified_eval_rows.csv}"
TRAIN_CSV="${SUCC_UNIFIED_JOINT_TRAIN_CSV:-$DATA_ROOT/unified_joint_train_rows.csv}"
EVAL_CSV="${SUCC_UNIFIED_JOINT_EVAL_CSV:-$DATA_ROOT/unified_joint_eval_rows.csv}"
MANIFEST_JSON="${SUCC_UNIFIED_JOINT_MANIFEST_JSON:-$DATA_ROOT/unified_joint_manifest.json}"
TRAIN_FEATURES_DIR="${SUCC_UNIFIED_JOINT_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_JOINT_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
BASE_CHECKPOINT="${SUCC_UNIFIED_JOINT_BASE_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"

case "$STAGE" in
u1)
  OUTPUT_DIR="${SUCC_UNIFIED_JOINT_OUTPUT_DIR:-$JOINT_ROOT/u1_joint_sft}"
  TEACHER_CHECKPOINT=""
  DISTILL_WEIGHT="${SUCC_UNIFIED_DISTILL_WEIGHT:-0}"
  ;;
u2)
  OUTPUT_DIR="${SUCC_UNIFIED_JOINT_OUTPUT_DIR:-$JOINT_ROOT/u2_joint_protected_sft}"
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
require_file "$BASE_CHECKPOINT"
require_dir "$TRAIN_FEATURES_DIR"
require_dir "$EVAL_FEATURES_DIR"
if [[ -n "$TEACHER_CHECKPOINT" ]]; then
  require_file "$TEACHER_CHECKPOINT"
fi

mkdir -p "$DATA_ROOT" "$OUTPUT_DIR"

echo "=== prepare leakage-audited joint rows ==="
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_unified_joint_rows.py" \
  --source-train-csv "$SOURCE_TRAIN_CSV" \
  --source-eval-csv "$SOURCE_EVAL_CSV" \
  --train-output-csv "$TRAIN_CSV" \
  --eval-output-csv "$EVAL_CSV" \
  --manifest-json "$MANIFEST_JSON" \
  --denovo-train-per-count "${SUCC_UNIFIED_JOINT_DENOVO_TRAIN_PER_COUNT:-2000}" \
  --denovo-eval-per-count "${SUCC_UNIFIED_JOINT_DENOVO_EVAL_PER_COUNT:-1000}" \
  --edit-train-per-task "${SUCC_UNIFIED_JOINT_EDIT_TRAIN_PER_TASK:-500}" \
  --edit-eval-per-task "${SUCC_UNIFIED_JOINT_EDIT_EVAL_PER_TASK:-100}" \
  --overlap-policy "${SUCC_UNIFIED_JOINT_OVERLAP_POLICY:-drop_train}" \
  --seed "${SUCC_UNIFIED_JOINT_DATA_SEED:-41}"

echo
echo "=== Unified Joint v2 training: stage=$STAGE ==="
SUCC_UNIFIED_TRAIN_CSV="$TRAIN_CSV" \
SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
SUCC_UNIFIED_OUTPUT_DIR="$OUTPUT_DIR" \
SUCC_UNIFIED_TRAIN_FEATURES_DIR="$TRAIN_FEATURES_DIR" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$EVAL_FEATURES_DIR" \
SUCC_UNIFIED_CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_compat}" \
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}" \
SUCC_UNIFIED_INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-with_image}" \
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
bash "$SCRIPT_DIR/run_unified_smiles_generator_train.sh"

echo
echo "Unified Joint v2 ready:"
echo "  stage=$STAGE"
echo "  checkpoint=$OUTPUT_DIR/unified_smiles_generator.pt"
echo "  manifest=$MANIFEST_JSON"
echo "  train_csv=$TRAIN_CSV"
echo "  eval_csv=$EVAL_CSV"
echo "  distill_weight=$DISTILL_WEIGHT"
