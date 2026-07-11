#!/usr/bin/env bash
# Unified Fair Protocol v1 U1: de novo SFT with direct_compat on unified de novo rows.
#
# Warm-starts model weights from Direct 2p7p SFT, resets optimizer/epoch history.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
FAIR_ROOT="${SUCC_UNIFIED_FAIR_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_fair_v1}"
SUITE_ROOT="${SUCC_UNIFIED_FAIR_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
P2P7P_ROOT="${SUCC_UNIFIED_FAIR_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
DATA_ROOT="${SUCC_UNIFIED_FAIR_DATA_ROOT:-$FAIR_ROOT/dataset}"

SOURCE_TRAIN_CSV="${SUCC_UNIFIED_FAIR_SOURCE_TRAIN_CSV:-$SUITE_ROOT/dataset/unified_train_rows.csv}"
SOURCE_EVAL_CSV="${SUCC_UNIFIED_FAIR_SOURCE_EVAL_CSV:-$SUITE_ROOT/dataset/unified_eval_rows.csv}"
TRAIN_CSV="${SUCC_UNIFIED_FAIR_TRAIN_CSV:-$DATA_ROOT/unified_train_de_novo_rows.csv}"
EVAL_CSV="${SUCC_UNIFIED_FAIR_EVAL_CSV:-$P2P7P_ROOT/denovo_2p7p_eval_rows.csv}"
OUTPUT_DIR="${SUCC_UNIFIED_FAIR_U1_OUTPUT_DIR:-$FAIR_ROOT/u1_denovo_sft}"

TRAIN_FEATURES_DIR="${SUCC_UNIFIED_FAIR_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_FAIR_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
DIRECT_SFT_CHECKPOINT="${SUCC_UNIFIED_FAIR_DIRECT_SFT_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"

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
require_file "$EVAL_CSV"
require_file "$DIRECT_SFT_CHECKPOINT"
require_dir "$TRAIN_FEATURES_DIR"
require_dir "$EVAL_FEATURES_DIR"

mkdir -p "$DATA_ROOT" "$OUTPUT_DIR"

echo "=== filter unified de novo train rows ==="
"$PYTHON_BIN" "$SCRIPT_DIR/filter_unified_smiles_rows.py" \
  --input-csv "$SOURCE_TRAIN_CSV" \
  --output-csv "$TRAIN_CSV" \
  --task-mode de_novo

echo
echo "=== Unified Fair v1 U1 de novo SFT ==="
SUCC_UNIFIED_TRAIN_CSV="$TRAIN_CSV" \
SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
SUCC_UNIFIED_OUTPUT_DIR="$OUTPUT_DIR" \
SUCC_UNIFIED_TRAIN_FEATURES_DIR="$TRAIN_FEATURES_DIR" \
SUCC_UNIFIED_EVAL_FEATURES_DIR="$EVAL_FEATURES_DIR" \
SUCC_UNIFIED_CONDITION_LAYOUT="${SUCC_UNIFIED_CONDITION_LAYOUT:-direct_compat}" \
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}" \
SUCC_UNIFIED_INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-with_image}" \
SUCC_UNIFIED_RESUME_CHECKPOINT="$DIRECT_SFT_CHECKPOINT" \
SUCC_UNIFIED_RESET_TRAINING_STATE=1 \
SUCC_UNIFIED_EPOCHS="${SUCC_UNIFIED_EPOCHS:-4}" \
SUCC_UNIFIED_BATCH_SIZE="${SUCC_UNIFIED_BATCH_SIZE:-64}" \
SUCC_UNIFIED_EVAL_BATCH_SIZE="${SUCC_UNIFIED_EVAL_BATCH_SIZE:-128}" \
SUCC_UNIFIED_LR="${SUCC_UNIFIED_LR:-3e-4}" \
SUCC_UNIFIED_EVAL_LIMIT="${SUCC_UNIFIED_EVAL_LIMIT:-1000}" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_train.sh"

echo
echo "Unified Fair v1 U1 training outputs:"
echo "  checkpoint=$OUTPUT_DIR/unified_smiles_generator.pt"
echo "  train_csv=$TRAIN_CSV"
echo "  eval_csv=$EVAL_CSV (eval_limit=${SUCC_UNIFIED_EVAL_LIMIT:-1000} during training)"
