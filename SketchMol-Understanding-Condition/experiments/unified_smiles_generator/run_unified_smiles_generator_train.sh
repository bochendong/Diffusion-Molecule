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
SEED="${SUCC_UNIFIED_SEED:-7}"
DEVICE="${SUCC_DEVICE:-${SUCC_UNIFIED_DEVICE:-auto}}"

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
  --seed "$SEED"
  --device "$DEVICE"
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
echo "  input_modality=${INPUT_MODALITY:-auto}"
echo "  decoding_mode=$DECODING_MODE"

"$PYTHON_BIN" "${args[@]}"
