#!/usr/bin/env bash
# Submit official GeLLMO-C C-MuMO reproduction jobs plus oracle/eval postprocess.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

TASKS="${SUCC_GELLMOC_TASKS:-BPQ,ELQ,ACEP,BDPQ,DHMQ,CDE,ABMP,BCMQ,BDEQ,HLMPQ}"
SETTINGS="${SUCC_GELLMOC_SETTINGS:-seen}"
RUN_POSTPROCESS="${SUCC_GELLMOC_RUN_POSTPROCESS:-1}"
OUTPUT_DIR="${SUCC_GELLMOC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_gellmoc_official_cmumo_v1}"
POST_JOB_NAME="${SUCC_GELLMOC_POST_SLURM_JOB_NAME:-succ-gellmoc-official-post}"
POST_ACCOUNT="${SUCC_GELLMOC_POST_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
POST_TIME="${SUCC_GELLMOC_POST_SLURM_TIME:-12:00:00}"
POST_CPUS="${SUCC_GELLMOC_POST_SLURM_CPUS:-4}"
POST_MEM="${SUCC_GELLMOC_POST_SLURM_MEM:-32G}"
POST_PARTITION="${SUCC_GELLMOC_POST_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"

mkdir -p "$LOG_DIR"

declare -a JOB_IDS=()

submit_task() {
  local task="$1"
  local setting="$2"
  echo
  echo "=== submitting official GeLLMO-C task=$task setting=$setting ==="
  local output
  if ! output="$(
    SUCC_GELLMOC_TASK="$task" \
    SUCC_GELLMOC_SETTING="$setting" \
    SUCC_GELLMOC_OUTPUT_DIR="$OUTPUT_DIR" \
    bash "$PROJECT_DIR/scripts/submit_external_gellmoc_official_task.sh" 2>&1
  )"; then
    echo "$output"
    echo "ERROR: failed to submit official GeLLMO-C task=$task setting=$setting" >&2
    exit 1
  fi
  echo "$output"
  local job_id
  job_id="$(echo "$output" | sed -n 's/.*external_gellmoc_official_task_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "ERROR: could not parse task job id for $task/$setting." >&2
    exit 1
  fi
  JOB_IDS+=("$job_id")
}

join_by_colon() {
  local IFS=:
  echo "$*"
}

echo "Official GeLLMO-C C-MuMO suite"
echo "  tasks=$TASKS"
echo "  settings=$SETTINGS"
echo "  output_dir=$OUTPUT_DIR"
echo "  run_postprocess=$RUN_POSTPROCESS"

IFS=',' read -r -a task_array <<< "$TASKS"
IFS=',' read -r -a setting_array <<< "$SETTINGS"
for task in "${task_array[@]}"; do
  task_trimmed="${task#"${task%%[![:space:]]*}"}"
  task_trimmed="${task_trimmed%"${task_trimmed##*[![:space:]]}"}"
  [[ -z "$task_trimmed" ]] && continue
  for setting in "${setting_array[@]}"; do
    setting_trimmed="${setting#"${setting%%[![:space:]]*}"}"
    setting_trimmed="${setting_trimmed%"${setting_trimmed##*[![:space:]]}"}"
    [[ -z "$setting_trimmed" ]] && continue
    submit_task "$task_trimmed" "$setting_trimmed"
  done
done

if [[ "$RUN_POSTPROCESS" != "1" ]]; then
  echo
  echo "Official GeLLMO-C tasks submitted without postprocess."
  echo "  task_jobs=$(join_by_colon "${JOB_IDS[@]}")"
  exit 0
fi

dependency="afterok:$(join_by_colon "${JOB_IDS[@]}")"
echo
echo "=== submitting official GeLLMO-C postprocess after $dependency ==="
SBATCH_ARGS=(
  --account="$POST_ACCOUNT"
  --job-name="$POST_JOB_NAME"
  --time="$POST_TIME"
  --mem="$POST_MEM"
  --cpus-per-task="$POST_CPUS"
  --dependency="$dependency"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL,SUCC_GELLMOC_TASKS="$TASKS",SUCC_GELLMOC_SETTINGS="$SETTINGS",SUCC_GELLMOC_OUTPUT_DIR="$OUTPUT_DIR"
)
if [[ -n "$POST_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$POST_PARTITION")
fi
post_output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_gellmoc_official_postprocess.sh'")"
echo "$post_output"
post_job_id="$(echo "$post_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$post_job_id" ]]; then
  echo "ERROR: failed to submit official GeLLMO-C postprocess." >&2
  exit 1
fi

echo
echo "Official GeLLMO-C C-MuMO suite submitted."
echo "  task_jobs=$(join_by_colon "${JOB_IDS[@]}")"
echo "  postprocess_job=$post_job_id"
echo "  output_dir=$OUTPUT_DIR"
