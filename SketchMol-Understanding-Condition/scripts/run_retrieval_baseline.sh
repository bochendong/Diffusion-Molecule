#!/usr/bin/env bash
# Run retrieval-style ablation baselines for the strict QED edit dataset.

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

BASELINE_CSV="${SUCC_BASELINE_CSV:-SketchMol-Understanding-Condition/outputs/qed_edit_dataset_5k_diverse_strict/baseline_variants.csv}"
OUTPUT_DIR="${SUCC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/retrieval_baseline_qed_strict}"
RIDGE_ALPHA="${SUCC_RIDGE_ALPHA:-10.0}"

echo "Running retrieval baseline"
echo "  baseline_csv=$BASELINE_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  ridge_alpha=$RIDGE_ALPHA"

python "$PROJECT_DIR/scripts/train_retrieval_baseline.py" \
  --baseline-variants-csv "$BASELINE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --ridge-alpha "$RIDGE_ALPHA"

echo "Retrieval baseline finished."
