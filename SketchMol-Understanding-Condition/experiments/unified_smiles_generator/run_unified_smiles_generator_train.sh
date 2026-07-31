#!/usr/bin/env bash
# Train the standalone unified SMILES generator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-}"
EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-}"
OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_v1}"
TRAIN_FEATURES_DIR="${SUCC_UNIFIED_TRAIN_FEATURES_DIR:-}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-$TRAIN_FEATURES_DIR}"
CONDITION_FEATURE_ARRAY="${SUCC_UNIFIED_CONDITION_FEATURE_ARRAY:-query_tokens}"
CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}"
CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-unified}"
INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-}"
CONDITION_DIM="${SUCC_UNIFIED_CONDITION_DIM:-256}"
EPOCHS="${SUCC_UNIFIED_EPOCHS:-8}"
BATCH_SIZE="${SUCC_UNIFIED_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${SUCC_UNIFIED_EVAL_BATCH_SIZE:-128}"
LR="${SUCC_UNIFIED_LR:-3e-4}"
WEIGHT_DECAY="${SUCC_UNIFIED_WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${SUCC_UNIFIED_GRAD_CLIP:-1.0}"
D_MODEL="${SUCC_UNIFIED_D_MODEL:-256}"
NUM_LAYERS="${SUCC_UNIFIED_NUM_LAYERS:-4}"
NUM_HEADS="${SUCC_UNIFIED_NUM_HEADS:-8}"
DIM_FEEDFORWARD="${SUCC_UNIFIED_DIM_FEEDFORWARD:-1024}"
MAX_SMILES_LENGTH="${SUCC_UNIFIED_MAX_SMILES_LENGTH:-160}"
MAX_SOURCE_TOKENS="${SUCC_UNIFIED_MAX_SOURCE_TOKENS:-96}"
DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-sample}"
NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-20}"
BEAM_SIZE="${SUCC_UNIFIED_BEAM_SIZE:-20}"
BEAM_EXPAND_SIZE="${SUCC_UNIFIED_BEAM_EXPAND_SIZE:-64}"
BEAM_LENGTH_PENALTY="${SUCC_UNIFIED_BEAM_LENGTH_PENALTY:-0.8}"
TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-40}"
MAX_CANDIDATES="${SUCC_UNIFIED_MAX_CANDIDATES:-0}"
SEED="${SUCC_UNIFIED_SEED:-7}"
DEVICE="${SUCC_DEVICE:-${SUCC_UNIFIED_DEVICE:-auto}}"
SAMPLING_MODE="${SUCC_UNIFIED_SAMPLING_MODE:-random}"
SAMPLES_PER_EPOCH="${SUCC_UNIFIED_SAMPLES_PER_EPOCH:-0}"
TEACHER_CHECKPOINT="${SUCC_UNIFIED_TEACHER_CHECKPOINT:-}"
DISTILL_WEIGHT="${SUCC_UNIFIED_DISTILL_WEIGHT:-0}"
DISTILL_TEMPERATURE="${SUCC_UNIFIED_DISTILL_TEMPERATURE:-1.0}"
DISTILL_CONTROL="${SUCC_UNIFIED_DISTILL_CONTROL:-fixed}"
DISTILL_TARGET_KL="${SUCC_UNIFIED_DISTILL_TARGET_KL:-0.02}"
DISTILL_DUAL_LR="${SUCC_UNIFIED_DISTILL_DUAL_LR:-0.5}"
DISTILL_MIN_WEIGHT="${SUCC_UNIFIED_DISTILL_MIN_WEIGHT:-0.0}"
DISTILL_MAX_WEIGHT="${SUCC_UNIFIED_DISTILL_MAX_WEIGHT:-2.0}"
SOURCE_ENCODER_LAYERS="${SUCC_UNIFIED_SOURCE_ENCODER_LAYERS:-2}"
SOURCE_RESIDUAL_SCALE="${SUCC_UNIFIED_SOURCE_RESIDUAL_SCALE:-1.0}"

if [[ -z "$TRAIN_CSV" ]]; then
  echo "ERROR: set SUCC_UNIFIED_TRAIN_CSV" >&2
  exit 2
fi

args=(
  "$SCRIPT_DIR/unified_smiles_generator.py"
  train
  --train-csv "$TRAIN_CSV"
  --output-dir "$OUTPUT_DIR"
  --condition-feature-array "$CONDITION_FEATURE_ARRAY"
  --condition-feature-variant "$CONDITION_FEATURE_VARIANT"
  --condition-layout "$CONDITION_LAYOUT"
  --condition-dim "$CONDITION_DIM"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP"
  --d-model "$D_MODEL"
  --num-layers "$NUM_LAYERS"
  --num-heads "$NUM_HEADS"
  --dim-feedforward "$DIM_FEEDFORWARD"
  --max-smiles-length "$MAX_SMILES_LENGTH"
  --max-source-tokens "$MAX_SOURCE_TOKENS"
  --decoding-mode "$DECODING_MODE"
  --num-samples "$NUM_SAMPLES"
  --beam-size "$BEAM_SIZE"
  --beam-expand-size "$BEAM_EXPAND_SIZE"
  --beam-length-penalty "$BEAM_LENGTH_PENALTY"
  --top-k-candidates "$TOP_K_CANDIDATES"
  --max-candidates "$MAX_CANDIDATES"
  --seed "$SEED"
  --device "$DEVICE"
  --sampling-mode "$SAMPLING_MODE"
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --distill-weight "$DISTILL_WEIGHT"
  --distill-temperature "$DISTILL_TEMPERATURE"
  --distill-control "$DISTILL_CONTROL"
  --distill-target-kl "$DISTILL_TARGET_KL"
  --distill-dual-lr "$DISTILL_DUAL_LR"
  --distill-min-weight "$DISTILL_MIN_WEIGHT"
  --distill-max-weight "$DISTILL_MAX_WEIGHT"
  --source-encoder-layers "$SOURCE_ENCODER_LAYERS"
  --source-residual-scale "$SOURCE_RESIDUAL_SCALE"
)

if [[ -n "$INPUT_MODALITY" ]]; then
  args+=(--input-modality "$INPUT_MODALITY")
fi
if [[ -n "$EVAL_CSV" ]]; then
  args+=(--eval-csv "$EVAL_CSV")
fi
if [[ -n "$TRAIN_FEATURES_DIR" ]]; then
  args+=(--condition-features-dir "$TRAIN_FEATURES_DIR")
fi
if [[ -n "$EVAL_FEATURES_DIR" ]]; then
  args+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
fi
if [[ -n "${SUCC_UNIFIED_RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume-checkpoint "$SUCC_UNIFIED_RESUME_CHECKPOINT")
fi
if [[ -n "$TEACHER_CHECKPOINT" ]]; then
  args+=(--teacher-checkpoint "$TEACHER_CHECKPOINT")
fi
if [[ "${SUCC_UNIFIED_INCLUDE_SOURCE_COPY_CANDIDATE:-0}" == "1" ]]; then
  args+=(--include-source-copy-candidate)
fi
if [[ "${SUCC_UNIFIED_RESET_TRAINING_STATE:-0}" == "1" ]]; then
  args+=(--reset-training-state)
fi
if [[ "${SUCC_UNIFIED_SOURCE_AWARE:-0}" == "1" ]]; then
  args+=(--source-aware)
fi
if [[ "${SUCC_UNIFIED_ALLOW_ARCHITECTURE_WARMSTART:-0}" == "1" ]]; then
  args+=(--allow-architecture-warmstart)
fi
if [[ -n "${SUCC_UNIFIED_LIMIT:-}" ]]; then
  args+=(--limit "$SUCC_UNIFIED_LIMIT")
fi
if [[ -n "${SUCC_UNIFIED_EVAL_LIMIT:-}" ]]; then
  args+=(--eval-limit "$SUCC_UNIFIED_EVAL_LIMIT")
fi

echo "Unified SMILES generator training"
echo "  train_csv=$TRAIN_CSV"
echo "  eval_csv=${EVAL_CSV:-none}"
echo "  output_dir=$OUTPUT_DIR"
echo "  train_features=${TRAIN_FEATURES_DIR:-fallback}"
echo "  eval_features=${EVAL_FEATURES_DIR:-fallback}"
echo "  condition_feature_variant=$CONDITION_FEATURE_VARIANT"
echo "  condition_layout=$CONDITION_LAYOUT"
echo "  input_modality=${INPUT_MODALITY:-auto}"
echo "  decoding_mode=$DECODING_MODE"
echo "  sampling_mode=$SAMPLING_MODE"
echo "  samples_per_epoch=$SAMPLES_PER_EPOCH"
echo "  teacher_checkpoint=${TEACHER_CHECKPOINT:-none}"
echo "  distill_weight=$DISTILL_WEIGHT"
echo "  distill_control=$DISTILL_CONTROL target_kl=$DISTILL_TARGET_KL"
echo "  source_aware=${SUCC_UNIFIED_SOURCE_AWARE:-0} source_encoder_layers=$SOURCE_ENCODER_LAYERS"

"$PYTHON_BIN" "${args[@]}"
