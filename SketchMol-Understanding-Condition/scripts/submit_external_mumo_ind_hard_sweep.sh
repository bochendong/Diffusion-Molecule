#!/usr/bin/env bash
# Submit MuMO IND-hard GraphEditDSL sweeps plus dependent oracle/re-eval jobs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_FILE="${SUCC_IND_HARD_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
TASKS="${SUCC_IND_HARD_TASKS:-BDP,BDPQ,DPQ,BDMQ}"
MAX_ROWS_PER_TASK="${SUCC_IND_HARD_MAX_ROWS_PER_TASK:-200}"
OLD_ORACLE="${SUCC_IND_HARD_MERGE_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
ORACLE_OUTPUT="${SUCC_IND_HARD_ORACLE_OUTPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_mumo_indhard_v1/generated_properties.csv}"
ORACLE_WORK_DIR="${SUCC_IND_HARD_ORACLE_WORK_DIR:-$(dirname "$ORACLE_OUTPUT")}"
RUN_ORACLE_AFTER="${SUCC_IND_HARD_RUN_ORACLE_AFTER:-1}"
RUN_REEVAL_AFTER="${SUCC_IND_HARD_RUN_REEVAL_AFTER:-1}"
TOP_K="${SUCC_IND_HARD_TOP_K_CANDIDATES:-20}"

declare -a JOB_IDS=()
declare -a OUTPUT_DIRS=()
declare -a CANDIDATE_CSVS=()
declare -a VARIANT_NAMES=()

submit_variant() {
  local variant="$1"
  local output_dir="$2"
  local job_name="$3"
  local selection_mode="$4"
  local local_success="$5"
  local beam="$6"
  local max_row="$7"
  local property_weight="$8"
  local distance_weight="$9"
  local similarity_weight="${10}"
  local similarity_bonus="${11}"
  local copy_penalty="${12}"

  echo
  echo "=== submitting MuMO IND-hard $variant ==="
  local output
  if ! output="$(
    SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE="$SOURCE_FILE" \
    SUCC_EXTERNAL_GRAPH_EDIT_SUITE=mumo \
    SUCC_EXTERNAL_GRAPH_EDIT_TASKS="$TASKS" \
    SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT=all \
    SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK="$MAX_ROWS_PER_TASK" \
    SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR="$output_dir" \
    SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME="$job_name" \
    SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE="$selection_mode" \
    SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION="$local_success" \
    SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE="$beam" \
    SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW="$max_row" \
    SUCC_EXTERNAL_GRAPH_EDIT_PROPERTY_WEIGHT="$property_weight" \
    SUCC_EXTERNAL_GRAPH_EDIT_DISTANCE_WEIGHT="$distance_weight" \
    SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_WEIGHT="$similarity_weight" \
    SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_BONUS="$similarity_bonus" \
    SUCC_EXTERNAL_GRAPH_EDIT_COPY_PENALTY="$copy_penalty" \
    SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES="$TOP_K" \
    SUCC_EXTERNAL_GRAPH_EDIT_BUILD_ORACLE_CSV=0 \
    bash "$PROJECT_DIR/scripts/submit_direct_smiles_external_mumo_graph_edit_heuristic_2step.sh" 2>&1
  )"; then
    echo "$output"
    echo "WARN: failed to submit MuMO IND-hard $variant; continuing." >&2
    return 0
  fi
  echo "$output"
  local job_id
  job_id="$(echo "$output" | sed -n 's/.*external_mumo_graph_edit_agent_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -n "$job_id" ]]; then
    JOB_IDS+=("$job_id")
    OUTPUT_DIRS+=("$output_dir")
    CANDIDATE_CSVS+=("$output_dir/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv")
    VARIANT_NAMES+=("$variant")
  else
    echo "WARN: could not parse job id for MuMO IND-hard $variant." >&2
  fi
}

join_by_colon() {
  local IFS=:
  echo "$*"
}

echo "MuMO IND-hard external benchmark sweep"
echo "  source_file=$SOURCE_FILE"
echo "  tasks=$TASKS"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  top_k=$TOP_K"
echo "  old_oracle=$OLD_ORACLE"
echo "  oracle_output=$ORACLE_OUTPUT"

submit_variant \
  balanced \
  SketchMol-Understanding-Condition/outputs/external_mumo_indhard_graph_edit_balanced_v1 \
  succ-mumo-indhard-balanced \
  similarity_first \
  0.85 \
  128 \
  4096 \
  90 \
  8 \
  100 \
  160 \
  10

submit_variant \
  property_first \
  SketchMol-Understanding-Condition/outputs/external_mumo_indhard_graph_edit_property_first_v1 \
  succ-mumo-indhard-property \
  score \
  0.5 \
  160 \
  4096 \
  160 \
  12 \
  35 \
  40 \
  18

submit_variant \
  wide_balanced \
  SketchMol-Understanding-Condition/outputs/external_mumo_indhard_graph_edit_wide_balanced_v1 \
  succ-mumo-indhard-wide \
  similarity_first \
  0.7 \
  192 \
  8192 \
  120 \
  10 \
  85 \
  120 \
  14

if [[ "$RUN_ORACLE_AFTER" != "1" || "${#JOB_IDS[@]}" -eq 0 ]]; then
  echo
  echo "MuMO IND-hard sweep submitted without dependent oracle."
  exit 0
fi

dependency="afterok:$(join_by_colon "${JOB_IDS[@]}")"
input_csv="$(IFS=,; echo "${CANDIDATE_CSVS[*]}")"
merge_csv=""
if [[ -n "$OLD_ORACLE" ]]; then
  merge_csv="$OLD_ORACLE"
fi

echo
echo "=== submitting MuMO IND-hard oracle after $dependency ==="
oracle_output="$(
  SUCC_ORACLE_SLURM_JOB_NAME=succ-mumo-indhard-oracle \
  SUCC_ORACLE_SLURM_DEPENDENCY="$dependency" \
  SUCC_ORACLE_SLURM_TIME="${SUCC_IND_HARD_ORACLE_SLURM_TIME:-12:00:00}" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_WORK_DIR" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_OUTPUT" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$merge_csv" \
  SUCC_ORACLE_INPUT_CSV="$input_csv" \
  bash "$PROJECT_DIR/scripts/submit_external_multiproperty_generated_oracle_pipeline.sh" 2>&1
)"
echo "$oracle_output"
oracle_job_id="$(echo "$oracle_output" | sed -n 's/.*external_oracle_build_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$oracle_job_id" || "$RUN_REEVAL_AFTER" != "1" ]]; then
  echo
  echo "MuMO IND-hard sweep submitted."
  echo "  graph_jobs=$(join_by_colon "${JOB_IDS[@]}")"
  echo "  oracle_job=${oracle_job_id:-none}"
  exit 0
fi

for index in "${!CANDIDATE_CSVS[@]}"; do
  variant="${VARIANT_NAMES[$index]}"
  output_dir="${OUTPUT_DIRS[$index]}"
  candidate_csv="${CANDIDATE_CSVS[$index]}"
  echo
  echo "=== submitting MuMO IND-hard $variant oracle re-eval ==="
  SUCC_EXTERNAL_REEVAL_SLURM_JOB_NAME="succ-mumo-indhard-${variant}-reeval" \
  SUCC_EXTERNAL_REEVAL_SLURM_DEPENDENCY="afterok:$oracle_job_id" \
  SUCC_EXTERNAL_REEVAL_PREDICTION_CSV="$candidate_csv" \
  SUCC_EXTERNAL_REEVAL_OUTPUT_DIR="$output_dir/benchmark_with_indhard_oracle_v1" \
  SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV="$ORACLE_OUTPUT" \
  SUCC_EXTERNAL_REEVAL_SOURCE_PROPERTIES_CSV="$ORACLE_OUTPUT" \
  SUCC_EXTERNAL_REEVAL_REPORT_TITLE="MuMO IND-hard $variant Oracle Re-eval" \
  bash "$PROJECT_DIR/scripts/submit_external_multiproperty_re_eval.sh" || true
done

echo
echo "MuMO IND-hard sweep submission finished."
echo "  graph_jobs=$(join_by_colon "${JOB_IDS[@]}")"
echo "  oracle_job=$oracle_job_id"
