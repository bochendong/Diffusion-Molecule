#!/usr/bin/env bash
# Build generated-property oracle CSV for MuMO/C-MuMO official evaluation.
#
# Pipeline:
#   1. Collect unique SMILES from prediction/row CSVs
#   2. Run ADMET-AI (GeLLMO official ADMET oracle)
#   3. Merge ADMET-AI + local/TDC properties via score_external_multiproperty_oracles.py
#
# Usage:
#   SUCC_ORACLE_INPUT_CSV=path/to/predictions.csv \
#   SUCC_ORACLE_OUTPUT_CSV=path/to/generated_properties.csv \
#   bash SketchMol-Understanding-Condition/scripts/run_external_multiproperty_generated_oracle_pipeline.sh
#
# After the CSV exists, rerun official suite with:
#   SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=$SUCC_ORACLE_OUTPUT_CSV \
#   bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh

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

# ADMET-AI install note (ComputeCanada): load rdkit module before pip install admet-ai.
# Recommended: bash SketchMol-Understanding-Condition/scripts/setup_admet_ai_venv.sh
# then point SUCC_ADMET_PYTHON_BIN to ~/.venvs/admet_ai/bin/python

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-$PYTHON_BIN}"
WORK_DIR="${SUCC_ORACLE_WORK_DIR:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1}"
INPUT_CSV="${SUCC_ORACLE_INPUT_CSV:-}"
OUTPUT_CSV="${SUCC_ORACLE_OUTPUT_CSV:-$WORK_DIR/generated_properties.csv}"
SMILES_COLUMNS="${SUCC_ORACLE_SMILES_COLUMNS:-generated_smiles,source_smiles,target_smiles,smiles,SMILES,canonical_smiles}"
MERGE_CSVS="${SUCC_ORACLE_MERGE_PROPERTIES_CSV:-}"
BATCH_SIZE="${SUCC_ORACLE_ADMET_BATCH_SIZE:-256}"
SKIP_ADMET="${SUCC_ORACLE_SKIP_ADMET:-0}"

if [[ -z "$INPUT_CSV" ]]; then
  echo "ERROR: set SUCC_ORACLE_INPUT_CSV to one or more comma-separated prediction CSV paths." >&2
  exit 2
fi

mkdir -p "$WORK_DIR"
UNIQUE_SMILES_CSV="$WORK_DIR/unique_smiles.csv"
ADMET_CSV="$WORK_DIR/admet_ai_properties.csv"

echo "External multiproperty oracle pipeline"
echo "  input_csv=$INPUT_CSV"
echo "  output_csv=$OUTPUT_CSV"
echo "  work_dir=$WORK_DIR"
echo "  python_bin=$PYTHON_BIN"
echo "  admet_python_bin=$ADMET_PYTHON_BIN"
echo "  merge_properties_csv=${MERGE_CSVS:-none}"
echo "  skip_admet=$SKIP_ADMET"

IFS=',' read -r -a input_paths <<< "$INPUT_CSV"
collect_args=(--output-csv "$UNIQUE_SMILES_CSV")
for path in "${input_paths[@]}"; do
  trimmed="${path#"${path%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "$trimmed" ]]; then
    collect_args+=(--input-csv "$trimmed")
  fi
done

"$PYTHON_BIN" "$PROJECT_DIR/scripts/collect_external_multiproperty_oracle_smiles.py" \
  "${collect_args[@]}" \
  --smiles-columns "$SMILES_COLUMNS"

if [[ "$SKIP_ADMET" != "1" ]]; then
  "$ADMET_PYTHON_BIN" "$PROJECT_DIR/scripts/predict_external_multiproperty_admet_ai.py" \
    --input-smiles-csv "$UNIQUE_SMILES_CSV" \
    --output-csv "$ADMET_CSV" \
    --batch-size "$BATCH_SIZE"
else
  echo "skip ADMET-AI prediction (SUCC_ORACLE_SKIP_ADMET=1)"
fi

score_args=(
  --output-csv "$OUTPUT_CSV"
  --report-json "${OUTPUT_CSV%.csv}.summary.json"
  --report-md "${OUTPUT_CSV%.csv}.report.md"
)
for path in "${input_paths[@]}"; do
  trimmed="${path#"${path%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "$trimmed" ]]; then
    score_args+=(--input-csv "$trimmed")
  fi
done
if [[ -s "$ADMET_CSV" ]]; then
  score_args+=(--merge-properties-csv "$ADMET_CSV")
fi
if [[ -n "$MERGE_CSVS" ]]; then
  IFS=',' read -r -a merge_paths <<< "$MERGE_CSVS"
  for path in "${merge_paths[@]}"; do
    trimmed="${path#"${path%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    if [[ -n "$trimmed" ]]; then
      score_args+=(--merge-properties-csv "$trimmed")
    fi
  done
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/score_external_multiproperty_oracles.py" "${score_args[@]}"

echo
echo "Done."
echo "  generated_properties_csv=$OUTPUT_CSV"
echo "  coverage_report=${OUTPUT_CSV%.csv}.report.md"
echo
echo "Rerun official suite with:"
echo "  SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=$OUTPUT_CSV \\"
echo "  bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh"
