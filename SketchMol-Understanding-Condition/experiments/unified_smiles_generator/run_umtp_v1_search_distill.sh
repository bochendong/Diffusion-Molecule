#!/usr/bin/env bash
# Search on a train-only balanced pool, select verifier-feasible molecules, and distill them into the policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${UMTP_TRAIN_SEED:-${SUCC_UNIFIED_SEED:-7}}"
JOINT_ROOT="${UMTP_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_v1}"
SEED_ROOT="$POLICY_ROOT/seed_${SEED}"

TRAIN_CSV="${UMTP_TRAIN_CSV:-$JOINT_ROOT/dataset/unified_joint_train_rows.csv}"
VALIDATION_CSV="${UMTP_VALIDATION_CSV:-$JOINT_ROOT/dataset/unified_joint_validation_rows.csv}"
TRAIN_FEATURES="${UMTP_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
VALIDATION_FEATURES="${UMTP_VALIDATION_FEATURES_DIR:-$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm}"
SFT_CHECKPOINT="${UMTP_SFT_CHECKPOINT:-$SEED_ROOT/sft/unified_smiles_generator.pt}"
SEARCH_ROOT="${UMTP_SEARCH_ROOT:-$SEED_ROOT/search_distillation}"
SEARCH_POOL="$SEARCH_ROOT/search_pool.csv"
SEARCH_CANDIDATES="$SEARCH_ROOT/candidate_pool/unified_smiles_candidate_predictions.csv"
DISTILL_ROWS="$SEARCH_ROOT/distillation_rows.csv"
DISTILL_OUTPUT="${UMTP_DISTILL_OUTPUT_DIR:-$SEED_ROOT/policy}"

for path in "$TRAIN_CSV" "$VALIDATION_CSV" "$SFT_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
mkdir -p "$SEARCH_ROOT"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_transformation_search_pool.py" \
  --input-csv "$TRAIN_CSV" \
  --output-csv "$SEARCH_POOL" \
  --manifest-json "$SEARCH_ROOT/search_pool_manifest.json" \
  --rows-per-group "${UMTP_SEARCH_ROWS_PER_GROUP:-100}" \
  --seed "${UMTP_SEARCH_POOL_SEED:-71}"

SUCC_UNIFIED_CHECKPOINT="$SFT_CHECKPOINT" \
SUCC_UNIFIED_EVAL_CSV="$SEARCH_POOL" \
SUCC_UNIFIED_OUTPUT_DIR="$SEARCH_ROOT/candidate_pool" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$TRAIN_FEATURES" \
SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
SUCC_UNIFIED_DECODING_MODE=sample \
SUCC_UNIFIED_NUM_SAMPLES="${UMTP_SEARCH_CANDIDATES:-64}" \
SUCC_UNIFIED_TOP_K_CANDIDATES="${UMTP_SEARCH_CANDIDATES:-64}" \
SUCC_UNIFIED_MAX_CANDIDATES="${UMTP_SEARCH_CANDIDATES:-64}" \
SUCC_UNIFIED_TEMPERATURE="${UMTP_SEARCH_TEMPERATURE:-0.85}" \
SUCC_UNIFIED_TOP_K="${UMTP_SEARCH_TOP_K:-40}" \
SUCC_UNIFIED_TOP_P="${UMTP_SEARCH_TOP_P:-0.95}" \
SUCC_UNIFIED_SEED="${UMTP_SEARCH_SEED:-101}" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_sample.sh"

"$PYTHON_BIN" "$SCRIPT_DIR/build_transformation_search_distillation_rows.py" \
  --source-rows-csv "$SEARCH_POOL" \
  --candidate-csv "$SEARCH_CANDIDATES" \
  --output-csv "$DISTILL_ROWS" \
  --manifest-json "$SEARCH_ROOT/distillation_manifest.json" \
  --min-property-success "${UMTP_DISTILL_MIN_PROPERTY_SUCCESS:-1.0}" \
  --min-edit-similarity "${UMTP_DISTILL_MIN_EDIT_SIMILARITY:-0.65}" \
  --winners-per-condition "${UMTP_DISTILL_WINNERS_PER_CONDITION:-1}" \
  --source-replay-ratio "${UMTP_DISTILL_SOURCE_REPLAY_RATIO:-1.0}" \
  --seed "${UMTP_DISTILL_SEED:-73}"

SUCC_UNIFIED_TRAIN_CSV="$DISTILL_ROWS" \
SUCC_UNIFIED_EVAL_CSV="$VALIDATION_CSV" \
SUCC_UNIFIED_OUTPUT_DIR="$DISTILL_OUTPUT" \
SUCC_UNIFIED_TRAIN_FEATURES_DIR="$TRAIN_FEATURES" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$VALIDATION_FEATURES" \
SUCC_UNIFIED_RESUME_CHECKPOINT="$SFT_CHECKPOINT" \
SUCC_UNIFIED_TEACHER_CHECKPOINT="$SFT_CHECKPOINT" \
SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
SUCC_UNIFIED_SOURCE_AWARE=1 \
SUCC_UNIFIED_RESET_TRAINING_STATE=1 \
SUCC_UNIFIED_SAMPLING_MODE=task_balanced \
SUCC_UNIFIED_SAMPLES_PER_EPOCH="${UMTP_DISTILL_SAMPLES_PER_EPOCH:-0}" \
SUCC_UNIFIED_EPOCHS="${UMTP_DISTILL_EPOCHS:-1}" \
SUCC_UNIFIED_BATCH_SIZE="${UMTP_BATCH_SIZE:-64}" \
SUCC_UNIFIED_LR="${UMTP_DISTILL_LR:-5e-5}" \
SUCC_UNIFIED_DISTILL_WEIGHT="${UMTP_DISTILL_RETENTION_INITIAL_WEIGHT:-0.1}" \
SUCC_UNIFIED_DISTILL_CONTROL=adaptive \
SUCC_UNIFIED_DISTILL_TARGET_KL="${UMTP_DISTILL_RETENTION_TARGET_KL:-0.01}" \
SUCC_UNIFIED_DISTILL_DUAL_LR="${UMTP_DISTILL_RETENTION_DUAL_LR:-0.5}" \
SUCC_UNIFIED_SEED="$SEED" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_train.sh"

echo "UMTP v1 search-distilled policy ready: $DISTILL_OUTPUT/unified_smiles_generator.pt"
