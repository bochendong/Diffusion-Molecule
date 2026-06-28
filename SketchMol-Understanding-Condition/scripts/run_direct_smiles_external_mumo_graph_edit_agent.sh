#!/usr/bin/env bash
# Run MuMO source-conditioned GraphEditDSL agent benchmark.

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
OUTPUT_DIR="${SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_graph_edit_agent_v1}"
BASE_AGENTIC_OUTPUT_DIR="${SUCC_EXTERNAL_GRAPH_EDIT_BASE_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1}"
SOURCE_FILE="${SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
ROWS_CSV="${SUCC_EXTERNAL_GRAPH_EDIT_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_rows.csv}"
SUMMARY_JSON="${SUCC_EXTERNAL_GRAPH_EDIT_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_rows.summary.json}"
TASK_SPEC_JSON="${SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPEC_JSON:-$OUTPUT_DIR/external_multiproperty_task_specs.json}"
DIRECT_PREDICTION_CSV="${SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV:-$BASE_AGENTIC_OUTPUT_DIR/direct_smiles_proposals.csv}"
BENCHMARK_OUTPUT_DIR="${SUCC_EXTERNAL_GRAPH_EDIT_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_graph_edit_agent}"
PREDICTION_CSV="${SUCC_EXTERNAL_GRAPH_EDIT_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/graph_edit_agent_predictions.csv}"
PLAN_JSONL="${SUCC_EXTERNAL_GRAPH_EDIT_PLAN_JSONL:-$OUTPUT_DIR/graph_edit_plans.jsonl}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_GRAPH_EDIT_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"

SUITE="${SUCC_EXTERNAL_GRAPH_EDIT_SUITE:-mumo}"
TASK_SPLIT="${SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT:-all}"
TASKS="${SUCC_EXTERNAL_GRAPH_EDIT_TASKS:-}"
INPUT_SPLIT="${SUCC_EXTERNAL_GRAPH_EDIT_INPUT_SPLIT:-all}"
MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK:-200}"
SEED="${SUCC_EXTERNAL_GRAPH_EDIT_SEED:-17}"
FORCE_EXPORT="${SUCC_EXTERNAL_GRAPH_EDIT_FORCE_EXPORT:-0}"
PLANNER_MODE="${SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE:-heuristic_graph_dsl}"
SELECTION_MODE="${SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE:-similarity_first}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_GRAPH_EDIT_MIN_SOURCE_TANIMOTO:-0.4}"
PLANNER_STEPS="${SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS:-1}"
BEAM_SIZE="${SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE:-64}"
SITE_LIMIT="${SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT:-32}"
MAX_PLANS_PER_PROPERTY="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY:-160}"
MAX_CANDIDATES_PER_PARENT="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT:-256}"
MAX_CANDIDATES_PER_ROW="${SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW:-4096}"
SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION="${SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION:-1.0}"
PROPERTY_WEIGHT="${SUCC_EXTERNAL_GRAPH_EDIT_PROPERTY_WEIGHT:-100}"
DISTANCE_WEIGHT="${SUCC_EXTERNAL_GRAPH_EDIT_DISTANCE_WEIGHT:-10}"
SIMILARITY_WEIGHT="${SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_WEIGHT:-30}"
SIMILARITY_BONUS="${SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_BONUS:-80}"
COPY_PENALTY="${SUCC_EXTERNAL_GRAPH_EDIT_COPY_PENALTY:-8}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$BENCHMARK_OUTPUT_DIR"

echo "MuMO GraphEditDSL agent benchmark"
echo "  python=$PYTHON_BIN"
echo "  source_file=$SOURCE_FILE"
echo "  output_dir=$OUTPUT_DIR"
echo "  direct_prediction_csv=$DIRECT_PREDICTION_CSV"
echo "  rows_csv=$ROWS_CSV"
echo "  planner_mode=$PLANNER_MODE"
echo "  selection_mode=$SELECTION_MODE"
echo "  planner_steps=$PLANNER_STEPS"
echo "  beam_size=$BEAM_SIZE"
echo "  site_limit=$SITE_LIMIT"
echo "  max_plans_per_property=$MAX_PLANS_PER_PROPERTY"
echo "  max_candidates_per_parent=$MAX_CANDIDATES_PER_PARENT"
echo "  max_candidates_per_row=$MAX_CANDIDATES_PER_ROW"
echo "  min_source_tanimoto=$MIN_SOURCE_TANIMOTO"

if [[ "$FORCE_EXPORT" == "1" || ! -f "$ROWS_CSV" ]]; then
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

if [[ ! -f "$DIRECT_PREDICTION_CSV" ]]; then
  echo "ERROR: missing direct proposal CSV: $DIRECT_PREDICTION_CSV" >&2
  echo "Run submit_direct_smiles_external_mumo_agentic_revise.sh first, or set SUCC_EXTERNAL_GRAPH_EDIT_DIRECT_PREDICTION_CSV." >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/build_external_graph_edit_agent_predictions.py" \
  --rows-csv "$ROWS_CSV" \
  --direct-prediction-csv "$DIRECT_PREDICTION_CSV" \
  --prediction-csv "$PREDICTION_CSV" \
  --plan-jsonl "$PLAN_JSONL" \
  --planner-mode "$PLANNER_MODE" \
  --selection-mode "$SELECTION_MODE" \
  --min-source-tanimoto "$MIN_SOURCE_TANIMOTO" \
  --planner-steps "$PLANNER_STEPS" \
  --beam-size "$BEAM_SIZE" \
  --site-limit "$SITE_LIMIT" \
  --max-plans-per-property "$MAX_PLANS_PER_PROPERTY" \
  --max-candidates-per-parent "$MAX_CANDIDATES_PER_PARENT" \
  --max-candidates-per-row "$MAX_CANDIDATES_PER_ROW" \
  --similarity-first-min-local-success-fraction "$SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION" \
  --property-weight "$PROPERTY_WEIGHT" \
  --distance-weight "$DISTANCE_WEIGHT" \
  --similarity-weight "$SIMILARITY_WEIGHT" \
  --similarity-bonus "$SIMILARITY_BONUS" \
  --copy-penalty "$COPY_PENALTY" \
  --seed "$SEED"

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
  --report-title "SUCC External MuMO GraphEditDSL Agent Benchmark" \
  "${EVAL_ARGS[@]}"

echo
echo "MuMO GraphEditDSL agent benchmark ready:"
echo "  rows=$ROWS_CSV"
echo "  direct_predictions=$DIRECT_PREDICTION_CSV"
echo "  graph_predictions=$PREDICTION_CSV"
echo "  plans=$PLAN_JSONL"
echo "  report=$BENCHMARK_OUTPUT_DIR/external_multiproperty_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/external_multiproperty_summary.csv"
