#!/usr/bin/env bash
# Build the first QED-improvement scaffold edit dataset on the server.

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

INPUT_CSV="${SUCC_INPUT_CSV:-SketchMol-MultiProperty-EditDataset/data/train_table.csv}"
OUTPUT_DIR="${SUCC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/qed_edit_dataset_5k_diverse_strict}"
LIMIT="${SUCC_LIMIT:-5000}"
MAX_PAIRS="${SUCC_MAX_PAIRS:-200}"

echo "Building QED scaffold edit dataset"
echo "  input_csv=$INPUT_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  limit=$LIMIT"
echo "  max_pairs=$MAX_PAIRS"

python "$PROJECT_DIR/scripts/build_edit_dataset.py" \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --property-name QED \
  --direction increase \
  --limit "$LIMIT" \
  --max-pairs "$MAX_PAIRS" \
  --max-pairs-per-scaffold 10 \
  --min-abs-delta 0.05 \
  --min-similarity 0.2 \
  --max-similarity 0.9 \
  --eval-fraction 0.2

python "$PROJECT_DIR/scripts/build_baseline_variants.py" \
  --edit-pairs-csv "$OUTPUT_DIR/edit_pairs.csv" \
  --output-csv "$OUTPUT_DIR/baseline_variants.csv"

echo "QED dataset smoke finished."
