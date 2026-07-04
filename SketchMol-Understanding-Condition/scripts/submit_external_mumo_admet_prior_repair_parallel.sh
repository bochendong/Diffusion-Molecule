#!/usr/bin/env bash
# Submit MuMO ADMET-prior GraphEdit repair as parallel per-task jobs + merge + oracle DAG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

SOURCE_FILE="${SUCC_MUMO_ADMET_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
TASK_SPLIT="${SUCC_MUMO_ADMET_TASK_SPLIT:-all}"
MAX_ROWS_PER_TASK="${SUCC_MUMO_ADMET_MAX_ROWS_PER_TASK:-200}"
OUTPUT_DIR="${SUCC_MUMO_ADMET_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_graph_edit_admet_prior_v1}"
OLD_ORACLE="${SUCC_MUMO_ADMET_MERGE_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
ORACLE_OUTPUT="${SUCC_MUMO_ADMET_ORACLE_OUTPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_mumo_admet_prior_v1/generated_properties.csv}"
ORACLE_WORK_DIR="${SUCC_MUMO_ADMET_ORACLE_WORK_DIR:-$(dirname "$ORACLE_OUTPUT")}"
TOP_K="${SUCC_MUMO_ADMET_TOP_K_CANDIDATES:-40}"
RUN_ORACLE_AFTER="${SUCC_MUMO_ADMET_RUN_ORACLE_AFTER:-1}"
RUN_REEVAL_AFTER="${SUCC_MUMO_ADMET_RUN_REEVAL_AFTER:-1}"
TASK_LIST="${SUCC_MUMO_ADMET_TASK_LIST:-BDP,BDQ,BPQ,DPQ,BDPQ,MPQ,BDMQ,BHMQ,BMPQ,HMPQ}"
GRAPH_TIME="${SUCC_MUMO_ADMET_GRAPH_SLURM_TIME:-8:00:00}"
GRAPH_MEM="${SUCC_MUMO_ADMET_GRAPH_SLURM_MEM:-64G}"
MERGE_TIME="${SUCC_MUMO_ADMET_MERGE_SLURM_TIME:-01:00:00}"
MERGE_MEM="${SUCC_MUMO_ADMET_MERGE_SLURM_MEM:-32G}"
CHECKPOINT_EVERY="${SUCC_MUMO_ADMET_CHECKPOINT_EVERY:-10}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_MUMO_ADMET_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
CPUS="${SUCC_MUMO_ADMET_GRAPH_SLURM_CPUS:-8}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"

join_by_colon() {
  local IFS=:
  echo "$*"
}

export_common_graph_env() {
  export SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE="$SOURCE_FILE"
  export SUCC_EXTERNAL_GRAPH_EDIT_SUITE=mumo
  export SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT="$TASK_SPLIT"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK="$MAX_ROWS_PER_TASK"
  export SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS="${SUCC_MUMO_ADMET_REQUIRE_DIRECT_PROPOSALS:-1}"
  export SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE=admet_prior_graph_dsl
  export SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE="${SUCC_MUMO_ADMET_SELECTION_MODE:-similarity_first}"
  export SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION="${SUCC_MUMO_ADMET_LOCAL_SUCCESS_FRACTION:-0.75}"
  export SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS="${SUCC_MUMO_ADMET_PLANNER_STEPS:-2}"
  export SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE="${SUCC_MUMO_ADMET_BEAM_SIZE:-192}"
  export SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT="${SUCC_MUMO_ADMET_SITE_LIMIT:-64}"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY="${SUCC_MUMO_ADMET_MAX_PLANS_PER_PROPERTY:-224}"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT="${SUCC_MUMO_ADMET_MAX_CANDIDATES_PER_PARENT:-224}"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW="${SUCC_MUMO_ADMET_MAX_CANDIDATES_PER_ROW:-8192}"
  export SUCC_EXTERNAL_GRAPH_EDIT_PROPERTY_WEIGHT="${SUCC_MUMO_ADMET_PROPERTY_WEIGHT:-100}"
  export SUCC_EXTERNAL_GRAPH_EDIT_DISTANCE_WEIGHT="${SUCC_MUMO_ADMET_DISTANCE_WEIGHT:-8}"
  export SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_WEIGHT="${SUCC_MUMO_ADMET_SIMILARITY_WEIGHT:-85}"
  export SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_BONUS="${SUCC_MUMO_ADMET_SIMILARITY_BONUS:-140}"
  export SUCC_EXTERNAL_GRAPH_EDIT_ADMET_PRIOR_WEIGHT="${SUCC_MUMO_ADMET_PRIOR_WEIGHT:-70}"
  export SUCC_EXTERNAL_GRAPH_EDIT_COPY_PENALTY="${SUCC_MUMO_ADMET_COPY_PENALTY:-10}"
  export SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES="$TOP_K"
  export SUCC_EXTERNAL_GRAPH_EDIT_BUILD_ORACLE_CSV=0
  export SUCC_EXTERNAL_GRAPH_EDIT_SLURM_TIME="$GRAPH_TIME"
  export SUCC_EXTERNAL_GRAPH_EDIT_SLURM_MEM="$GRAPH_MEM"
  export SUCC_EXTERNAL_GRAPH_EDIT_SLURM_CPUS="$CPUS"
  export SUCC_EXTERNAL_GRAPH_EDIT_GPU_PROFILE=none
  export SUCC_EXTERNAL_GRAPH_EDIT_CHECKPOINT=1
  export SUCC_EXTERNAL_GRAPH_EDIT_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
  export SUCC_EXTERNAL_GRAPH_EDIT_RESUME=1
}

echo "MuMO ADMET-prior GraphEdit repair (parallel tasks)"
echo "  source_file=$SOURCE_FILE"
echo "  task_list=$TASK_LIST"
echo "  task_split=$TASK_SPLIT"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  output_dir=$OUTPUT_DIR"
echo "  graph_time=$GRAPH_TIME"
echo "  graph_mem=$GRAPH_MEM"
echo "  checkpoint_every=$CHECKPOINT_EVERY"
echo "  old_oracle=$OLD_ORACLE"
echo "  oracle_output=$ORACLE_OUTPUT"
echo "  top_k=$TOP_K"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

ROWS_CSV="$OUTPUT_DIR/external_multiproperty_rows.csv"
if [[ ! -f "$ROWS_CSV" ]]; then
  echo
  echo "=== exporting full MuMO rows once for merge ordering ==="
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
    --source-file "$SOURCE_FILE" \
    --output-csv "$ROWS_CSV" \
    --summary-json "$OUTPUT_DIR/external_multiproperty_rows.summary.json" \
    --task-spec-json "$OUTPUT_DIR/external_multiproperty_task_specs.json" \
    --suite mumo \
    --task-split "$TASK_SPLIT" \
    --tasks all \
    --input-split all \
    --max-rows-per-task "$MAX_ROWS_PER_TASK" \
    --seed 17
fi

declare -a GRAPH_JOB_IDS=()
declare -a TASK_NAMES=()
IFS=',' read -r -a TASK_ARRAY <<< "$TASK_LIST"
printf '%s\n' "${TASK_ARRAY[@]}" > "$OUTPUT_DIR/parallel_task_list.txt"
export_common_graph_env

for task in "${TASK_ARRAY[@]}"; do
  task="$(echo "$task" | xargs)"
  [[ -n "$task" ]] || continue
  task_output="$OUTPUT_DIR/tasks/$task"
  echo
  echo "=== submitting MuMO ADMET-prior graph task=$task ==="
  graph_output="$(
    SUCC_EXTERNAL_GRAPH_EDIT_TASKS="$task" \
    SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR="$task_output" \
    SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME="succ-mumo-admet-${task,,}" \
    bash "$PROJECT_DIR/scripts/submit_direct_smiles_external_mumo_graph_edit_heuristic_2step.sh" 2>&1
  )"
  echo "$graph_output"
  job_id="$(echo "$graph_output" | sed -n 's/.*external_mumo_graph_edit_agent_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "ERROR: failed to submit graph job for task=$task" >&2
    exit 1
  fi
  GRAPH_JOB_IDS+=("$job_id")
  TASK_NAMES+=("$task")
done

if [[ ${#GRAPH_JOB_IDS[@]} -eq 0 ]]; then
  echo "ERROR: no graph jobs submitted." >&2
  exit 1
fi

graph_dependency="afterok:$(join_by_colon "${GRAPH_JOB_IDS[@]}")"
echo
echo "=== submitting MuMO ADMET-prior shard merge ==="
merge_output="$(
  sbatch \
    --account="$ACCOUNT" \
    --job-name="succ-mumo-admet-merge" \
    --time="$MERGE_TIME" \
    --mem="$MERGE_MEM" \
    --cpus-per-task=2 \
    --dependency="$graph_dependency" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL,SUCC_MUMO_ADMET_OUTPUT_DIR="$OUTPUT_DIR" \
    --wrap="bash '$PROJECT_DIR/scripts/run_external_mumo_admet_prior_merge_task_shards.sh'" 2>&1
)"
echo "$merge_output"
merge_job_id="$(echo "$merge_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$merge_job_id" ]]; then
  echo "ERROR: failed to submit shard merge job." >&2
  exit 1
fi

candidate_csv="$OUTPUT_DIR/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv"
if [[ "$RUN_ORACLE_AFTER" != "1" ]]; then
  echo
  echo "MuMO ADMET-prior parallel graph submitted without dependent oracle."
  echo "  graph_jobs=$(join_by_colon "${GRAPH_JOB_IDS[@]}")"
  echo "  merge_job=$merge_job_id"
  echo "  candidate_csv=$candidate_csv"
  exit 0
fi

merge_csv=""
if [[ -n "$OLD_ORACLE" ]]; then
  merge_csv="$OLD_ORACLE"
fi

echo
echo "=== submitting MuMO ADMET-prior oracle ==="
oracle_output="$(
  SUCC_ORACLE_SLURM_JOB_NAME="${SUCC_MUMO_ADMET_ORACLE_JOB_NAME:-succ-mumo-admet-prior-oracle}" \
  SUCC_ORACLE_SLURM_DEPENDENCY="afterok:$merge_job_id" \
  SUCC_ORACLE_SLURM_TIME="${SUCC_MUMO_ADMET_ORACLE_SLURM_TIME:-12:00:00}" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_WORK_DIR" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_OUTPUT" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$merge_csv" \
  SUCC_ORACLE_INPUT_CSV="$candidate_csv" \
  bash "$PROJECT_DIR/scripts/submit_external_multiproperty_generated_oracle_pipeline.sh" 2>&1
)"
echo "$oracle_output"
oracle_job_id="$(echo "$oracle_output" | sed -n 's/.*external_oracle_build_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$oracle_job_id" || "$RUN_REEVAL_AFTER" != "1" ]]; then
  echo
  echo "MuMO ADMET-prior parallel sweep submitted."
  echo "  graph_jobs=$(join_by_colon "${GRAPH_JOB_IDS[@]}")"
  echo "  merge_job=$merge_job_id"
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
echo "MuMO ADMET-prior parallel repair submitted."
echo "  tasks=${TASK_NAMES[*]}"
echo "  graph_jobs=$(join_by_colon "${GRAPH_JOB_IDS[@]}")"
echo "  merge_job=$merge_job_id"
echo "  oracle_job=$oracle_job_id"
echo "  output_dir=$OUTPUT_DIR"
echo "  candidate_csv=$candidate_csv"
echo "  oracle_csv=$ORACLE_OUTPUT"
