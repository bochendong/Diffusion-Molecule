#!/usr/bin/env bash
# Sample selected/candidate SMILES from a standalone unified generator checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
CHECKPOINT="${SUCC_UNIFIED_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_v1/unified_smiles_generator.pt}"
EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-}"
OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_v1/sample}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-${SUCC_UNIFIED_FEATURES_DIR:-}}"
CONDITION_FEATURE_ARRAY="${SUCC_UNIFIED_CONDITION_FEATURE_ARRAY:-query_tokens}"
CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}"
INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-}"
CONDITION_DIM="${SUCC_UNIFIED_CONDITION_DIM:-256}"
MAX_SMILES_LENGTH="${SUCC_UNIFIED_MAX_SMILES_LENGTH:-160}"
MAX_SOURCE_TOKENS="${SUCC_UNIFIED_MAX_SOURCE_TOKENS:-96}"
DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-sample}"
NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-20}"
BEAM_SIZE="${SUCC_UNIFIED_BEAM_SIZE:-20}"
BEAM_EXPAND_SIZE="${SUCC_UNIFIED_BEAM_EXPAND_SIZE:-64}"
BEAM_LENGTH_PENALTY="${SUCC_UNIFIED_BEAM_LENGTH_PENALTY:-0.8}"
TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-40}"
TEMPERATURE="${SUCC_UNIFIED_TEMPERATURE:-0.7}"
TOP_K="${SUCC_UNIFIED_TOP_K:-24}"
TOP_P="${SUCC_UNIFIED_TOP_P:-0.9}"
SEED="${SUCC_UNIFIED_SEED:-7}"
DEVICE="${SUCC_DEVICE:-${SUCC_UNIFIED_DEVICE:-auto}}"

if [[ -z "$EVAL_CSV" ]]; then
  echo "ERROR: set SUCC_UNIFIED_EVAL_CSV" >&2
  exit 2
fi

args=(
  "$SCRIPT_DIR/unified_smiles_generator.py"
  sample
  --checkpoint "$CHECKPOINT"
  --eval-csv "$EVAL_CSV"
  --output-dir "$OUTPUT_DIR"
  --condition-feature-array "$CONDITION_FEATURE_ARRAY"
  --condition-feature-variant "$CONDITION_FEATURE_VARIANT"
  --condition-dim "$CONDITION_DIM"
  --max-smiles-length "$MAX_SMILES_LENGTH"
  --max-source-tokens "$MAX_SOURCE_TOKENS"
  --decoding-mode "$DECODING_MODE"
  --num-samples "$NUM_SAMPLES"
  --beam-size "$BEAM_SIZE"
  --beam-expand-size "$BEAM_EXPAND_SIZE"
  --beam-length-penalty "$BEAM_LENGTH_PENALTY"
  --top-k-candidates "$TOP_K_CANDIDATES"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ -n "$INPUT_MODALITY" ]]; then
  args+=(--input-modality "$INPUT_MODALITY")
fi
if [[ -n "$EVAL_FEATURES_DIR" ]]; then
  args+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
fi
if [[ -n "${SUCC_UNIFIED_EVAL_LIMIT:-}" ]]; then
  args+=(--eval-limit "$SUCC_UNIFIED_EVAL_LIMIT")
fi

echo "Unified SMILES generator sampling"
echo "  checkpoint=$CHECKPOINT"
echo "  eval_csv=$EVAL_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  eval_features=${EVAL_FEATURES_DIR:-fallback}"
echo "  condition_feature_variant=$CONDITION_FEATURE_VARIANT"
echo "  input_modality=${INPUT_MODALITY:-auto}"
echo "  decoding_mode=$DECODING_MODE"

"$PYTHON_BIN" "${args[@]}"
