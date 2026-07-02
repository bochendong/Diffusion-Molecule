#!/usr/bin/env bash
# Submit a MuMO repair sweep with ADMET-prior GraphEditDSL, then oracle/re-eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_FILE="${SUCC_MUMO_ADMET_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
TASKS="${SUCC_MUMO_ADMET_TASKS:-}"
TASK_SPLIT="${SUCC_MUMO_ADMET_TASK_SPLIT:-all}"
MAX_ROWS_PER_TASK="${SUCC_MUMO_ADMET_MAX_ROWS_PER_TASK:-200}"
OUTPUT_DIR="${SUCC_MUMO_ADMET_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_graph_edit_admet_prior_v1}"
OLD_ORACLE="${SUCC_MUMO_ADMET_MERGE_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
ORACLE_OUTPUT="${SUCC_MUMO_ADMET_ORACLE_OUTPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_mumo_admet_prior_v1/generated_properties.csv}"
ORACLE_WORK_DIR="${SUCC_MUMO_ADMET_ORACLE_WORK_DIR:-$(dirname "$ORACLE_OUTPUT")}"
TOP_K="${SUCC_MUMO_ADMET_TOP_K_CANDIDATES:-40}"
RUN_ORACLE_AFTER="${SUCC_MUMO_ADMET_RUN_ORACLE_AFTER:-1}"
RUN_REEVAL_AFTER="${SUCC_MUMO_ADMET_RUN_REEVAL_AFTER:-1}"

echo "MuMO ADMET-prior GraphEdit repair"
echo "  source_file=$SOURCE_FILE"
echo "  tasks=${TASKS:-all}"
echo "  task_split=$TASK_SPLIT"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  output_dir=$OUTPUT_DIR"
echo "  old_oracle=$OLD_ORACLE"
echo "  oracle_output=$ORACLE_OUTPUT"
echo "  top_k=$TOP_K"

if ! graph_output="$(
  SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE="$SOURCE_FILE" \
  SUCC_EXTERNAL_GRAPH_EDIT_SUITE=mumo \
  SUCC_EXTERNAL_GRAPH_EDIT_TASKS="$TASKS" \
  SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT="$TASK_SPLIT" \
  SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK="$MAX_ROWS_PER_TASK" \
  SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR="$OUTPUT_DIR" \
  SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME="${SUCC_MUMO_ADMET_GRAPH_JOB_NAME:-succ-mumo-admet-prior-graph}" \
  SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS="${SUCC_MUMO_ADMET_REQUIRE_DIRECT_PROPOSALS:-1}" \
  SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE=admet_prior_graph_dsl \
  SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE="${SUCC_MUMO_ADMET_SELECTION_MODE:-similarity_first}" \
  SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION="${SUCC_MUMO_ADMET_LOCAL_SUCCESS_FRACTION:-0.75}" \
  SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS="${SUCC_MUMO_ADMET_PLANNER_STEPS:-2}" \
  SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE="${SUCC_MUMO_ADMET_BEAM_SIZE:-192}" \
  SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT="${SUCC_MUMO_ADMET_SITE_LIMIT:-64}" \
  SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY="${SUCC_MUMO_ADMET_MAX_PLANS_PER_PROPERTY:-224}" \
  SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT="${SUCC_MUMO_ADMET_MAX_CANDIDATES_PER_PARENT:-224}" \
  SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW="${SUCC_MUMO_ADMET_MAX_CANDIDATES_PER_ROW:-8192}" \
  SUCC_EXTERNAL_GRAPH_EDIT_PROPERTY_WEIGHT="${SUCC_MUMO_ADMET_PROPERTY_WEIGHT:-100}" \
  SUCC_EXTERNAL_GRAPH_EDIT_DISTANCE_WEIGHT="${SUCC_MUMO_ADMET_DISTANCE_WEIGHT:-8}" \
  SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_WEIGHT="${SUCC_MUMO_ADMET_SIMILARITY_WEIGHT:-85}" \
  SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_BONUS="${SUCC_MUMO_ADMET_SIMILARITY_BONUS:-140}" \
  SUCC_EXTERNAL_GRAPH_EDIT_ADMET_PRIOR_WEIGHT="${SUCC_MUMO_ADMET_PRIOR_WEIGHT:-70}" \
  SUCC_EXTERNAL_GRAPH_EDIT_COPY_PENALTY="${SUCC_MUMO_ADMET_COPY_PENALTY:-10}" \
  SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES="$TOP_K" \
  SUCC_EXTERNAL_GRAPH_EDIT_BUILD_ORACLE_CSV=0 \
  SUCC_EXTERNAL_GRAPH_EDIT_SLURM_TIME="${SUCC_MUMO_ADMET_GRAPH_SLURM_TIME:-24:00:00}" \
  SUCC_EXTERNAL_GRAPH_EDIT_SLURM_MEM="${SUCC_MUMO_ADMET_GRAPH_SLURM_MEM:-128G}" \
  bash "$PROJECT_DIR/scripts/submit_direct_smiles_external_mumo_graph_edit_heuristic_2step.sh" 2>&1
)"; then
  echo "$graph_output"
  echo "ERROR: failed to submit MuMO ADMET-prior graph job." >&2
  exit 1
fi
echo "$graph_output"
graph_job_id="$(echo "$graph_output" | sed -n 's/.*external_mumo_graph_edit_agent_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$graph_job_id" ]]; then
  echo "ERROR: could not parse ADMET-prior graph job id." >&2
  exit 1
fi

candidate_csv="$OUTPUT_DIR/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv"
if [[ "$RUN_ORACLE_AFTER" != "1" ]]; then
  echo
  echo "MuMO ADMET-prior graph submitted without dependent oracle."
  echo "  graph_job=$graph_job_id"
  echo "  candidate_csv=$candidate_csv"
  exit 0
fi

merge_csv=""
if [[ -n "$OLD_ORACLE" ]]; then
  merge_csv="$OLD_ORACLE"
fi

echo
echo "=== submitting MuMO ADMET-prior oracle ==="
if ! oracle_output="$(
  SUCC_ORACLE_SLURM_JOB_NAME="${SUCC_MUMO_ADMET_ORACLE_JOB_NAME:-succ-mumo-admet-prior-oracle}" \
  SUCC_ORACLE_SLURM_DEPENDENCY="afterok:$graph_job_id" \
  SUCC_ORACLE_SLURM_TIME="${SUCC_MUMO_ADMET_ORACLE_SLURM_TIME:-12:00:00}" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_WORK_DIR" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_OUTPUT" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$merge_csv" \
  SUCC_ORACLE_INPUT_CSV="$candidate_csv" \
  bash "$PROJECT_DIR/scripts/submit_external_multiproperty_generated_oracle_pipeline.sh" 2>&1
)"; then
  echo "$oracle_output"
  echo "ERROR: failed to submit MuMO ADMET-prior oracle job." >&2
  exit 1
fi
echo "$oracle_output"
oracle_job_id="$(echo "$oracle_output" | sed -n 's/.*external_oracle_build_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$oracle_job_id" || "$RUN_REEVAL_AFTER" != "1" ]]; then
  echo
  echo "MuMO ADMET-prior sweep submitted."
  echo "  graph_job=$graph_job_id"
  echo "  oracle_job=${oracle_job_id:-none}"
  exit 0
fi

echo
echo "=== submitting MuMO ADMET-prior oracle re-eval ==="
SUCC_EXTERNAL_REEVAL_SLURM_JOB_NAME="${SUCC_MUMO_ADMET_REEVAL_JOB_NAME:-succ-mumo-admet-prior-reeval}" \
SUCC_EXTERNAL_REEVAL_SLURM_DEPENDENCY="afterok:$oracle_job_id" \
SUCC_EXTERNAL_REEVAL_PREDICTION_CSV="$candidate_csv" \
SUCC_EXTERNAL_REEVAL_OUTPUT_DIR="$OUTPUT_DIR/benchmark_with_admet_prior_oracle_v1" \
SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV="$ORACLE_OUTPUT" \
SUCC_EXTERNAL_REEVAL_SOURCE_PROPERTIES_CSV="$ORACLE_OUTPUT" \
bash "$PROJECT_DIR/scripts/submit_external_multiproperty_re_eval.sh" || true

echo
echo "MuMO ADMET-prior repair submitted."
echo "  graph_job=$graph_job_id"
echo "  oracle_job=$oracle_job_id"
echo "  output_dir=$OUTPUT_DIR"
echo "  candidate_csv=$candidate_csv"
echo "  oracle_csv=$ORACLE_OUTPUT"
