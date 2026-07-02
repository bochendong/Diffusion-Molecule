#!/usr/bin/env bash
# Re-evaluate a run and build an aligned comparison/failure-diagnosis report.

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
LABEL="${SUCC_EXTERNAL_ALIGNED_LABEL:-ours}"
OUTPUT_DIR="${SUCC_EXTERNAL_ALIGNED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_multiproperty_aligned_eval_v1}"
EVAL_DIR="${SUCC_EXTERNAL_ALIGNED_EVAL_DIR:-$OUTPUT_DIR/eval_${LABEL}}"
REPORT_DIR="${SUCC_EXTERNAL_ALIGNED_REPORT_DIR:-$OUTPUT_DIR/aligned_report}"
PREDICTION_CSV="${SUCC_EXTERNAL_ALIGNED_PREDICTION_CSV:-${SUCC_EXTERNAL_REEVAL_PREDICTION_CSV:-}}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_ALIGNED_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_ALIGNED_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_REEVAL_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}}"
SMILES_COLUMN="${SUCC_EXTERNAL_ALIGNED_SMILES_COLUMN:-generated_smiles}"
SOURCE_SMILES_COLUMN="${SUCC_EXTERNAL_ALIGNED_SOURCE_SMILES_COLUMN:-source_smiles}"
GROUP_COLUMN="${SUCC_EXTERNAL_ALIGNED_GROUP_COLUMN:-condition_id}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_ALIGNED_MIN_SOURCE_TANIMOTO:-0.4}"
REPORT_TITLE="${SUCC_EXTERNAL_ALIGNED_REPORT_TITLE:-External Multi-property Aligned Eval: $LABEL}"
COMPARE_RUNS="${SUCC_EXTERNAL_ALIGNED_COMPARE_RUNS:-}"

if [[ -z "$SOURCE_PROPERTIES_CSV" && -n "$GENERATED_PROPERTIES_CSV" ]]; then
  SOURCE_PROPERTIES_CSV="$GENERATED_PROPERTIES_CSV"
fi
if [[ -z "$PREDICTION_CSV" || ! -f "$PREDICTION_CSV" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_ALIGNED_PREDICTION_CSV to an existing candidate prediction CSV." >&2
  exit 2
fi
if [[ -z "$GENERATED_PROPERTIES_CSV" || ! -f "$GENERATED_PROPERTIES_CSV" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_ALIGNED_GENERATED_PROPERTIES_CSV to an existing oracle CSV." >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$EVAL_DIR" "$REPORT_DIR"

echo "External multi-property aligned eval"
echo "  label=$LABEL"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  eval_dir=$EVAL_DIR"
echo "  report_dir=$REPORT_DIR"
echo "  generated_properties_csv=$GENERATED_PROPERTIES_CSV"
echo "  source_properties_csv=${SOURCE_PROPERTIES_CSV:-none}"
echo "  group_column=$GROUP_COLUMN"
echo "  compare_runs=${COMPARE_RUNS:-$LABEL=$EVAL_DIR}"

EVAL_ARGS=(
  --prediction-csv "$PREDICTION_CSV"
  --output-dir "$EVAL_DIR"
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

if [[ -z "$COMPARE_RUNS" ]]; then
  COMPARE_RUNS="$LABEL=$EVAL_DIR"
fi
summary_args=()
IFS=',' read -r -a compare_array <<< "$COMPARE_RUNS"
for run_spec in "${compare_array[@]}"; do
  trimmed="${run_spec#"${run_spec%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "$trimmed" ]]; then
    summary_args+=(--run "$trimmed")
  fi
done

"$PYTHON_BIN" "$PROJECT_DIR/scripts/summarize_external_multiproperty_aligned_runs.py" \
  "${summary_args[@]}" \
  --output-dir "$REPORT_DIR" \
  --group-column "$GROUP_COLUMN"

echo
echo "External multi-property aligned eval ready:"
echo "  eval_report=$EVAL_DIR/external_multiproperty_report.md"
echo "  aligned_report=$REPORT_DIR/external_multiproperty_aligned_report.md"
echo "  comparison_csv=$REPORT_DIR/external_multiproperty_aligned_comparison.csv"
echo "  failure_csv=$REPORT_DIR/external_multiproperty_failure_breakdown.csv"
