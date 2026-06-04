#!/usr/bin/env bash
# Run objective-direction classifier ablations on the mixed-objective dataset.

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
OUTPUT_DIR="${SUCC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/objective_classifier_mixed_v2}"
CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-}"
FEATURE_ARRAY="${SUCC_FEATURE_ARRAY:-pooled}"

echo "Running objective-direction classifier"
echo "  baseline_csv=$BASELINE_CSV"
echo "  output_dir=$OUTPUT_DIR"
if [[ -n "$CONDITION_FEATURES_DIR" ]]; then
  echo "  condition_features_dir=$CONDITION_FEATURES_DIR"
  echo "  feature_array=$FEATURE_ARRAY"
fi

FEATURE_ARGS=()
if [[ -n "$CONDITION_FEATURES_DIR" ]]; then
  FEATURE_ARGS=(--condition-features-dir "$CONDITION_FEATURES_DIR" --feature-array "$FEATURE_ARRAY")
fi

python "$PROJECT_DIR/scripts/train_objective_classifier.py" \
  --baseline-variants-csv "$BASELINE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  "${FEATURE_ARGS[@]}" \
  --epochs 400 \
  --learning-rate 0.1 \
  --l2 0.001

echo "Objective-direction classifier finished."
