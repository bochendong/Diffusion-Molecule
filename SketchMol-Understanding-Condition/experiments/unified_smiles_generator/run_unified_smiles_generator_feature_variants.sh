#!/usr/bin/env bash
# Prepare full/text-only condition-feature variants for the unified generator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-}"
EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-}"
OUTPUT_ROOT="${SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_feature_variants_v1}"
VARIANTS="${SUCC_UNIFIED_FEATURE_VARIANTS:-full,text_only}"
RUN_TRAIN_EXPORT="${SUCC_UNIFIED_FEATURE_VARIANTS_RUN_TRAIN:-1}"
RUN_EVAL_EXPORT="${SUCC_UNIFIED_FEATURE_VARIANTS_RUN_EVAL:-1}"
ENCODER="${SUCC_UNIFIED_FEATURE_ENCODER:-hf_vlm}"
POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"

if [[ "$RUN_TRAIN_EXPORT" == "1" && -z "$TRAIN_CSV" ]]; then
  echo "ERROR: set SUCC_UNIFIED_TRAIN_CSV when train feature export is enabled." >&2
  exit 2
fi
if [[ "$RUN_EVAL_EXPORT" == "1" && -z "$EVAL_CSV" ]]; then
  echo "ERROR: set SUCC_UNIFIED_EVAL_CSV when eval feature export is enabled." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

export_one() {
  local split="$1"
  local source_csv="$2"
  local prepared_csv="$OUTPUT_ROOT/${split}_feature_variant_rows.csv"
  local feature_dir="$OUTPUT_ROOT/${split}_condition_features_${ENCODER}"

  echo
  echo "=== preparing $split feature variants ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_condition_feature_variants.py" \
    --input-csv "$source_csv" \
    --output-csv "$prepared_csv" \
    --variants "$VARIANTS" \
    --default-split "$split"

  echo
  echo "=== exporting $split condition features ==="
  SUCC_ENCODER="$ENCODER" \
  SUCC_VARIANTS="$VARIANTS" \
  SUCC_BASELINE_CSV="$prepared_csv" \
  SUCC_OUTPUT_DIR="$feature_dir" \
  SUCC_POOLED_DIM="$POOLED_DIM" \
  SUCC_NUM_QUERIES="$NUM_QUERIES" \
  SUCC_QUERY_DIM="$QUERY_DIM" \
  SUCC_HF_BATCH_SIZE="$HF_BATCH_SIZE" \
  bash "$PROJECT_DIR/scripts/run_condition_encoder_export.sh"

  echo "  ${split}_prepared_csv=$prepared_csv"
  echo "  ${split}_features_dir=$feature_dir"
}

echo "Unified SMILES condition-feature variants"
echo "  train_csv=${TRAIN_CSV:-none}"
echo "  eval_csv=${EVAL_CSV:-none}"
echo "  output_root=$OUTPUT_ROOT"
echo "  variants=$VARIANTS"
echo "  encoder=$ENCODER"
echo "  hf_batch_size=$HF_BATCH_SIZE"

if [[ "$RUN_TRAIN_EXPORT" == "1" ]]; then
  export_one "train" "$TRAIN_CSV"
fi
if [[ "$RUN_EVAL_EXPORT" == "1" ]]; then
  export_one "eval" "$EVAL_CSV"
fi

echo
echo "Feature variants ready:"
echo "  output_root=$OUTPUT_ROOT"
echo "  train_features=$OUTPUT_ROOT/train_condition_features_${ENCODER}"
echo "  eval_features=$OUTPUT_ROOT/eval_condition_features_${ENCODER}"
