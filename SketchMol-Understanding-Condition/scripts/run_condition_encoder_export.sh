#!/usr/bin/env bash
# Export condition features/query tokens for mixed-objective baseline rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023
module load python/3.11
module load rdkit/2025.09.4

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BASELINE_CSV="${SUCC_BASELINE_CSV:-SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv}"
ENCODER="${SUCC_ENCODER:-proxy}"
OUTPUT_DIR="${SUCC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_${ENCODER}}"
LIMIT="${SUCC_LIMIT:-}"
IMAGE_ENCODER_CHECKPOINT="${SUCC_IMAGE_ENCODER_CHECKPOINT:-}"

echo "Exporting condition encoder features"
echo "  baseline_csv=$BASELINE_CSV"
echo "  encoder=$ENCODER"
echo "  output_dir=$OUTPUT_DIR"
if [[ -n "$LIMIT" ]]; then
  echo "  limit=$LIMIT"
fi
if [[ -n "$IMAGE_ENCODER_CHECKPOINT" ]]; then
  echo "  image_encoder_checkpoint=$IMAGE_ENCODER_CHECKPOINT"
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
CHECKPOINT_ARGS=()
if [[ -n "$IMAGE_ENCODER_CHECKPOINT" ]]; then
  CHECKPOINT_ARGS=(--image-encoder-checkpoint "$IMAGE_ENCODER_CHECKPOINT")
fi

python "$PROJECT_DIR/scripts/export_condition_features.py" \
  --encoder "$ENCODER" \
  --baseline-variants-csv "$BASELINE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  "${CHECKPOINT_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

echo "Condition encoder export finished."
