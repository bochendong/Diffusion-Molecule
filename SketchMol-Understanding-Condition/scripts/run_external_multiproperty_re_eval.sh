#!/usr/bin/env bash
# Re-evaluate an external multi-property prediction CSV with an oracle property CSV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PREDICTION_CSV="${SUCC_EXTERNAL_REEVAL_PREDICTION_CSV:-}"
OUTPUT_DIR="${SUCC_EXTERNAL_REEVAL_OUTPUT_DIR:-}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_REEVAL_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"
SMILES_COLUMN="${SUCC_EXTERNAL_REEVAL_SMILES_COLUMN:-generated_smiles}"
SOURCE_SMILES_COLUMN="${SUCC_EXTERNAL_REEVAL_SOURCE_SMILES_COLUMN:-source_smiles}"
GROUP_COLUMN="${SUCC_EXTERNAL_REEVAL_GROUP_COLUMN:-condition_id}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_REEVAL_MIN_SOURCE_TANIMOTO:-0.4}"
REPORT_TITLE="${SUCC_EXTERNAL_REEVAL_REPORT_TITLE:-External Multi-property Oracle Re-eval}"

if [[ -z "$SOURCE_PROPERTIES_CSV" && -n "$GENERATED_PROPERTIES_CSV" ]]; then
  SOURCE_PROPERTIES_CSV="$GENERATED_PROPERTIES_CSV"
fi
if [[ -z "$PREDICTION_CSV" || ! -f "$PREDICTION_CSV" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_REEVAL_PREDICTION_CSV to an existing prediction CSV." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_REEVAL_OUTPUT_DIR." >&2
  exit 2
fi
if [[ -z "$GENERATED_PROPERTIES_CSV" || ! -f "$GENERATED_PROPERTIES_CSV" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV to an existing oracle CSV." >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"

echo "External multi-property oracle re-eval"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  generated_properties_csv=$GENERATED_PROPERTIES_CSV"
echo "  source_properties_csv=${SOURCE_PROPERTIES_CSV:-none}"
echo "  group_column=$GROUP_COLUMN"
echo "  min_source_tanimoto=$MIN_SOURCE_TANIMOTO"

EVAL_ARGS=(
  --prediction-csv "$PREDICTION_CSV"
  --output-dir "$OUTPUT_DIR"
  --smiles-column "$SMILES_COLUMN"
  --source-smiles-column "$SOURCE_SMILES_COLUMN"
  --group-column "$GROUP_COLUMN"
  --min-source-tanimoto "$MIN_SOURCE_TANIMOTO"
  --report-title "$REPORT_TITLE"
  --generated-properties-csv "$GENERATED_PROPERTIES_CSV"
)
if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" "${EVAL_ARGS[@]}"

echo
echo "External multi-property oracle re-eval ready:"
echo "  report=$OUTPUT_DIR/external_multiproperty_report.md"
echo "  summary=$OUTPUT_DIR/external_multiproperty_summary.csv"
