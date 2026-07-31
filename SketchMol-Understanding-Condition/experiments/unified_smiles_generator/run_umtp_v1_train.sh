#!/usr/bin/env bash
# Train Unified Molecular Transformation Policy v1 from the audited Joint v2 corpus.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${UMTP_TRAIN_SEED:-${SUCC_UNIFIED_SEED:-7}}"
JOINT_ROOT="${UMTP_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
DIRECT_ROOT="${UMTP_DIRECT_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_v1}"

TRAIN_CSV="${UMTP_TRAIN_CSV:-$JOINT_ROOT/dataset/unified_joint_train_rows.csv}"
VALIDATION_CSV="${UMTP_VALIDATION_CSV:-$JOINT_ROOT/dataset/unified_joint_validation_rows.csv}"
TRAIN_FEATURES="${UMTP_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
VALIDATION_FEATURES="${UMTP_VALIDATION_FEATURES_DIR:-$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm}"
BASE_CHECKPOINT="${UMTP_BASE_CHECKPOINT:-$DIRECT_ROOT/direct_smiles_model/direct_smiles_generator.pt}"
OUTPUT_DIR="${UMTP_SFT_OUTPUT_DIR:-$POLICY_ROOT/seed_${SEED}/sft}"

for path in "$TRAIN_CSV" "$VALIDATION_CSV" "$BASE_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$VALIDATION_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing required feature directory: $path" >&2; exit 2; }
done

SUCC_UNIFIED_TRAIN_CSV="$TRAIN_CSV" \
SUCC_UNIFIED_EVAL_CSV="$VALIDATION_CSV" \
SUCC_UNIFIED_OUTPUT_DIR="$OUTPUT_DIR" \
SUCC_UNIFIED_TRAIN_FEATURES_DIR="$TRAIN_FEATURES" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$VALIDATION_FEATURES" \
SUCC_UNIFIED_RESUME_CHECKPOINT="$BASE_CHECKPOINT" \
SUCC_UNIFIED_TEACHER_CHECKPOINT="$BASE_CHECKPOINT" \
SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
SUCC_UNIFIED_SOURCE_AWARE=1 \
SUCC_UNIFIED_ALLOW_ARCHITECTURE_WARMSTART=1 \
SUCC_UNIFIED_RESET_TRAINING_STATE=1 \
SUCC_UNIFIED_SOURCE_ENCODER_LAYERS="${UMTP_SOURCE_ENCODER_LAYERS:-2}" \
SUCC_UNIFIED_SOURCE_RESIDUAL_SCALE="${UMTP_SOURCE_RESIDUAL_SCALE:-1.0}" \
SUCC_UNIFIED_SAMPLING_MODE=task_balanced \
SUCC_UNIFIED_SAMPLES_PER_EPOCH="${UMTP_SAMPLES_PER_EPOCH:-24000}" \
SUCC_UNIFIED_EPOCHS="${UMTP_SFT_EPOCHS:-4}" \
SUCC_UNIFIED_BATCH_SIZE="${UMTP_BATCH_SIZE:-64}" \
SUCC_UNIFIED_EVAL_BATCH_SIZE="${UMTP_EVAL_BATCH_SIZE:-128}" \
SUCC_UNIFIED_LR="${UMTP_SFT_LR:-1e-4}" \
SUCC_UNIFIED_DISTILL_WEIGHT="${UMTP_RETENTION_INITIAL_WEIGHT:-0.3}" \
SUCC_UNIFIED_DISTILL_CONTROL=adaptive \
SUCC_UNIFIED_DISTILL_TARGET_KL="${UMTP_RETENTION_TARGET_KL:-0.02}" \
SUCC_UNIFIED_DISTILL_DUAL_LR="${UMTP_RETENTION_DUAL_LR:-0.5}" \
SUCC_UNIFIED_DISTILL_MIN_WEIGHT="${UMTP_RETENTION_MIN_WEIGHT:-0.0}" \
SUCC_UNIFIED_DISTILL_MAX_WEIGHT="${UMTP_RETENTION_MAX_WEIGHT:-2.0}" \
SUCC_UNIFIED_SEED="$SEED" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_train.sh"

echo "UMTP v1 SFT ready: $OUTPUT_DIR/unified_smiles_generator.pt"
