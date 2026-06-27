#!/usr/bin/env bash
# Run a source-copy sanity check for external source-conditioned benchmarks.

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
OUTPUT_DIR="${SUCC_EXTERNAL_SOURCE_COPY_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_source_copy_sanity}"
SOURCE_FILE="${SUCC_EXTERNAL_SOURCE_COPY_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}}"
ROWS_CSV="${SUCC_EXTERNAL_SOURCE_COPY_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_rows.csv}"
TASK_SPEC_JSON="${SUCC_EXTERNAL_SOURCE_COPY_TASK_SPEC_JSON:-$OUTPUT_DIR/external_multiproperty_task_specs.json}"
SUMMARY_JSON="${SUCC_EXTERNAL_SOURCE_COPY_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_rows.summary.json}"
BENCHMARK_OUTPUT_DIR="${SUCC_EXTERNAL_SOURCE_COPY_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_source_copy}"
PREDICTION_CSV="${SUCC_EXTERNAL_SOURCE_COPY_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/source_copy_predictions.csv}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_SOURCE_COPY_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_SOURCE_COPY_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"

SUITE="${SUCC_EXTERNAL_SOURCE_COPY_SUITE:-${SUCC_EXTERNAL_MULTIPROP_SUITE:-mumo}}"
TASK_SPLIT="${SUCC_EXTERNAL_SOURCE_COPY_TASK_SPLIT:-${SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT:-all}}"
TASKS="${SUCC_EXTERNAL_SOURCE_COPY_TASKS:-${SUCC_EXTERNAL_MULTIPROP_TASKS:-}}"
INPUT_SPLIT="${SUCC_EXTERNAL_SOURCE_COPY_INPUT_SPLIT:-${SUCC_EXTERNAL_MULTIPROP_INPUT_SPLIT:-all}}"
MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_SOURCE_COPY_MAX_ROWS_PER_TASK:-${SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK:-200}}"
SEED="${SUCC_EXTERNAL_SOURCE_COPY_SEED:-${SUCC_EXTERNAL_MULTIPROP_SEED:-17}}"
FORCE_EXPORT="${SUCC_EXTERNAL_SOURCE_COPY_FORCE_EXPORT:-${SUCC_EXTERNAL_MULTIPROP_FORCE_EXPORT:-0}}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_SOURCE_COPY_MIN_SOURCE_TANIMOTO:-${SUCC_EXTERNAL_MULTIPROP_MIN_SOURCE_TANIMOTO:-0.4}}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$BENCHMARK_OUTPUT_DIR"

echo "External multi-property source-copy sanity"
echo "  python=$PYTHON_BIN"
echo "  source_file=$SOURCE_FILE"
echo "  output_dir=$OUTPUT_DIR"
echo "  suite=$SUITE"
echo "  task_split=$TASK_SPLIT"
echo "  tasks=${TASKS:-all}"
echo "  input_split=$INPUT_SPLIT"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"

if [[ "$FORCE_EXPORT" == "1" || ! -f "$ROWS_CSV" ]]; then
  if [[ -z "$SOURCE_FILE" || ! -f "$SOURCE_FILE" ]]; then
    echo "ERROR: set SUCC_EXTERNAL_SOURCE_COPY_SOURCE_FILE to a MuMO/C-MuMO JSON/JSONL/CSV source file." >&2
    exit 2
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
    --source-file "$SOURCE_FILE" \
    --output-csv "$ROWS_CSV" \
    --summary-json "$SUMMARY_JSON" \
    --task-spec-json "$TASK_SPEC_JSON" \
    --suite "$SUITE" \
    --task-split "$TASK_SPLIT" \
    --tasks "$TASKS" \
    --input-split "$INPUT_SPLIT" \
    --max-rows-per-task "$MAX_ROWS_PER_TASK" \
    --seed "$SEED"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/build_external_source_copy_predictions.py" \
  --rows-csv "$ROWS_CSV" \
  --prediction-csv "$PREDICTION_CSV"

EVAL_ARGS=()
if [[ -n "$GENERATED_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--generated-properties-csv "$GENERATED_PROPERTIES_CSV")
fi
if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
  --prediction-csv "$PREDICTION_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --smiles-column generated_smiles \
  --source-smiles-column source_smiles \
  --min-source-tanimoto "$MIN_SOURCE_TANIMOTO" \
  --report-title "External Multi-property Source-copy Sanity" \
  "${EVAL_ARGS[@]}"

echo
echo "External source-copy sanity ready:"
echo "  rows=$ROWS_CSV"
echo "  predictions=$PREDICTION_CSV"
echo "  report=$BENCHMARK_OUTPUT_DIR/external_multiproperty_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/external_multiproperty_summary.csv"
