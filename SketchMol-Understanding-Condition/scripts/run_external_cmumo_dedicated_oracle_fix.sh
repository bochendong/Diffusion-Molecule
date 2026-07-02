#!/usr/bin/env bash
# Build a C-MuMO-specific external-property oracle and audit source coverage.

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
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-$PYTHON_BIN}"
SOURCE_FILE="${SUCC_CMUMO_ORACLE_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/cmumo/test.json}"
WORK_DIR="${SUCC_CMUMO_ORACLE_WORK_DIR:-SketchMol-Understanding-Condition/outputs/external_oracle_build_cmumo_v1}"
OUTPUT_CSV="${SUCC_CMUMO_ORACLE_OUTPUT_CSV:-$WORK_DIR/generated_properties.csv}"
EXPORT_ROWS_CSV="${SUCC_CMUMO_ORACLE_EXPORT_ROWS_CSV:-$WORK_DIR/cmumo_oracle_source_rows.csv}"
SUMMARY_JSON="${SUCC_CMUMO_ORACLE_EXPORT_SUMMARY_JSON:-$WORK_DIR/cmumo_oracle_source_rows.summary.json}"
TASK_SPEC_JSON="${SUCC_CMUMO_ORACLE_TASK_SPEC_JSON:-$WORK_DIR/cmumo_oracle_task_specs.json}"
OLD_ORACLE="${SUCC_CMUMO_ORACLE_MERGE_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_sourcefix_v1/generated_properties.csv}"
FALLBACK_ORACLE="${SUCC_CMUMO_ORACLE_FALLBACK_MERGE_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
MAX_ROWS_PER_TASK="${SUCC_CMUMO_ORACLE_MAX_ROWS_PER_TASK:-${SUCC_OFFICIAL_MAX_ROWS_PER_TASK:-200}}"
SEED="${SUCC_CMUMO_ORACLE_SEED:-17}"
ADMET_BATCH_SIZE="${SUCC_CMUMO_ORACLE_ADMET_BATCH_SIZE:-${SUCC_ORACLE_ADMET_BATCH_SIZE:-256}}"
ADMET_REQUIRED_PROPERTIES="${SUCC_CMUMO_ORACLE_ADMET_REQUIRED_PROPERTIES:-${SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES:-ampa,bbbp,carc,erg,hia,liver,mutagenicity}}"
SKIP_ADMET="${SUCC_CMUMO_ORACLE_SKIP_ADMET:-${SUCC_ORACLE_SKIP_ADMET:-0}}"
EXTRA_INPUT_CSV="${SUCC_CMUMO_ORACLE_EXTRA_INPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_cmumo_official_copy_sanity_v1/external_multiproperty_rows.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_copy_sanity_v1/benchmark_source_copy/source_copy_predictions.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_copy_sanity_v1/benchmark_target_copy/target_copy_predictions.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/external_multiproperty_rows.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv}"

mkdir -p "$WORK_DIR"

echo "C-MuMO dedicated oracle fix"
echo "  source_file=$SOURCE_FILE"
echo "  work_dir=$WORK_DIR"
echo "  output_csv=$OUTPUT_CSV"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  old_oracle=$OLD_ORACLE"
echo "  fallback_oracle=$FALLBACK_ORACLE"
echo "  extra_input_csv=${EXTRA_INPUT_CSV:-none}"

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "ERROR: missing C-MuMO source file: $SOURCE_FILE" >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
  --source-file "$SOURCE_FILE" \
  --output-csv "$EXPORT_ROWS_CSV" \
  --summary-json "$SUMMARY_JSON" \
  --task-spec-json "$TASK_SPEC_JSON" \
  --suite cmumo \
  --task-split all \
  --input-split all \
  --max-rows-per-task "$MAX_ROWS_PER_TASK" \
  --seed "$SEED" \
  --respect-input-task

input_csv="$EXPORT_ROWS_CSV"
IFS=',' read -r -a extra_paths <<< "$EXTRA_INPUT_CSV"
for path in "${extra_paths[@]}"; do
  trimmed="${path#"${path%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "$trimmed" && -f "$trimmed" ]]; then
    input_csv="$input_csv,$trimmed"
  elif [[ -n "$trimmed" ]]; then
    echo "WARN: skipping missing extra input CSV: $trimmed" >&2
  fi
done

merge_csvs=""
if [[ -f "$OLD_ORACLE" ]]; then
  merge_csvs="$OLD_ORACLE"
elif [[ -f "$FALLBACK_ORACLE" ]]; then
  merge_csvs="$FALLBACK_ORACLE"
fi

SUCC_PYTHON_BIN="$PYTHON_BIN" \
SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
SUCC_ORACLE_INPUT_CSV="$input_csv" \
SUCC_ORACLE_OUTPUT_CSV="$OUTPUT_CSV" \
SUCC_ORACLE_WORK_DIR="$WORK_DIR" \
SUCC_ORACLE_MERGE_PROPERTIES_CSV="$merge_csvs" \
SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="$ADMET_REQUIRED_PROPERTIES" \
SUCC_ORACLE_ADMET_BATCH_SIZE="$ADMET_BATCH_SIZE" \
SUCC_ORACLE_SKIP_ADMET="$SKIP_ADMET" \
bash "$PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"

audit_args=(
  --oracle-properties-csv "$OUTPUT_CSV"
  --rows-csv "$EXPORT_ROWS_CSV"
  --output-md "$WORK_DIR/oracle_coverage_audit.md"
  --output-json "$WORK_DIR/oracle_coverage_audit.json"
)
for path in "${extra_paths[@]}"; do
  trimmed="${path#"${path%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  if [[ -n "$trimmed" && -f "$trimmed" ]]; then
    audit_args+=(--prediction-csv "$trimmed")
  fi
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/audit_external_multiproperty_oracle_coverage.py" "${audit_args[@]}"

echo
echo "C-MuMO dedicated oracle ready:"
echo "  generated_properties_csv=$OUTPUT_CSV"
echo "  exported_rows=$EXPORT_ROWS_CSV"
echo "  coverage_audit=$WORK_DIR/oracle_coverage_audit.md"
