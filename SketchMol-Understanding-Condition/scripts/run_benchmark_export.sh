#!/usr/bin/env bash
# Export understanding-condition predictions and materialize benchmark metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
BASELINE_VARIANTS_CSV="${SUCC_BASELINE_VARIANTS_CSV:-SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv}"
VARIANT="${SUCC_VARIANT:-full}"
EXPORT_DIR="${SUCC_BENCHMARK_EXPORT_DIR:-SketchMol-Understanding-Condition/outputs/benchmark_export_${VARIANT}}"
PREDICTIONS_CSV="${SUCC_BENCHMARK_PREDICTIONS_CSV:-$EXPORT_DIR/benchmark_predictions.csv}"
BENCHMARK_OUTPUT_DIR="${SUCC_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/understanding_condition_${VARIANT}}"
BENCHMARK_NAME="${SUCC_BENCHMARK_NAME:-understanding_condition_${VARIANT}_ridge}"
FINGERPRINT_BITS="${SUCC_FINGERPRINT_BITS:-512}"
TEXT_DIM="${SUCC_TEXT_DIM:-128}"
RANDOM_DIM="${SUCC_RANDOM_DIM:-128}"
RIDGE_ALPHA="${SUCC_RIDGE_ALPHA:-1.0}"
EVAL_SPLIT="${SUCC_EVAL_SPLIT:-eval}"
CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-}"

if [[ ! -f "$BASELINE_VARIANTS_CSV" ]]; then
  echo "ERROR: baseline variants CSV not found: $BASELINE_VARIANTS_CSV" >&2
  exit 2
fi

mkdir -p "$EXPORT_DIR"

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchMolBenchmark${PYTHONPATH:+:$PYTHONPATH}"

echo "Understanding-Condition benchmark export"
echo "  python=$PYTHON_BIN"
echo "  baseline_variants=$BASELINE_VARIANTS_CSV"
echo "  variant=$VARIANT"
echo "  predictions=$PREDICTIONS_CSV"
echo "  benchmark_output=$BENCHMARK_OUTPUT_DIR"
if [[ -n "$CONDITION_FEATURES_DIR" ]]; then
  echo "  condition_features=$CONDITION_FEATURES_DIR"
fi

FEATURE_ARGS=()
if [[ -n "$CONDITION_FEATURES_DIR" ]]; then
  FEATURE_ARGS=(--condition-features-dir "$CONDITION_FEATURES_DIR")
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/export_benchmark_predictions.py" \
  --baseline-variants-csv "$BASELINE_VARIANTS_CSV" \
  --output-csv "$PREDICTIONS_CSV" \
  --variant "$VARIANT" \
  --fingerprint-bits "$FINGERPRINT_BITS" \
  --text-dim "$TEXT_DIM" \
  --random-dim "$RANDOM_DIM" \
  --ridge-alpha "$RIDGE_ALPHA" \
  --eval-split "$EVAL_SPLIT" \
  "${FEATURE_ARGS[@]}"

SKETCHMOL_DIRECT_PREDICTIONS_CSV="$PREDICTIONS_CSV" \
SKETCHMOL_DIRECT_SMILES_COLUMN=generated_smiles \
SKETCHMOL_DIRECT_IMAGE_COLUMN=image_path \
SKETCHMOL_BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR" \
SKETCHMOL_BENCHMARK_NAME="$BENCHMARK_NAME" \
SKETCHMOL_BENCHMARK_PYTHON_BIN="$PYTHON_BIN" \
bash SketchMolBenchmark/scripts/materialize_direct_predictions.sh

echo
echo "Understanding-Condition benchmark ready:"
echo "  predictions=$PREDICTIONS_CSV"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
