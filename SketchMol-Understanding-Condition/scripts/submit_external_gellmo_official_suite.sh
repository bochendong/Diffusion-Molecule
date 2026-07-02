#!/usr/bin/env bash
# Submit official GeLLMO MuMO reproduction jobs plus oracle/eval postprocess.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

TASKS="${SUCC_GELLMO_TASKS:-bbbp+drd2+plogp,bbbp+drd2+qed,bbbp+plogp+qed,drd2+plogp+qed,bbbp+drd2+plogp+qed,mutagenicity+plogp+qed,bbbp+drd2+mutagenicity+qed,bbbp+hia+mutagenicity+qed,bbbp+mutagenicity+plogp+qed,hia+mutagenicity+plogp+qed}"
SETTINGS="${SUCC_GELLMO_SETTINGS:-seen}"
RUN_POSTPROCESS="${SUCC_GELLMO_RUN_POSTPROCESS:-1}"
OUTPUT_DIR="${SUCC_GELLMO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_gellmo_official_mumo_v1}"
POST_JOB_NAME="${SUCC_GELLMO_POST_SLURM_JOB_NAME:-succ-gellmo-official-post}"
POST_ACCOUNT="${SUCC_GELLMO_POST_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
POST_TIME="${SUCC_GELLMO_POST_SLURM_TIME:-12:00:00}"
POST_CPUS="${SUCC_GELLMO_POST_SLURM_CPUS:-4}"
POST_MEM="${SUCC_GELLMO_POST_SLURM_MEM:-32G}"
POST_PARTITION="${SUCC_GELLMO_POST_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"

mkdir -p "$LOG_DIR"

declare -a JOB_IDS=()

submit_task() {
  local task="$1"
  local setting="$2"
  echo
  echo "=== submitting official GeLLMO task=$task setting=$setting ==="
  local output
  if ! output="$(
    SUCC_GELLMO_TASK="$task" \
    SUCC_GELLMO_SETTING="$setting" \
    SUCC_GELLMO_OUTPUT_DIR="$OUTPUT_DIR" \
    bash "$PROJECT_DIR/scripts/submit_external_gellmo_official_task.sh" 2>&1
  )"; then
    echo "$output"
    echo "ERROR: failed to submit official GeLLMO task=$task setting=$setting" >&2
    exit 1
  fi
  echo "$output"
  local job_id
  job_id="$(echo "$output" | sed -n 's/.*external_gellmo_official_task_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
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

echo "Official GeLLMO MuMO suite"
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
  echo "Official GeLLMO tasks submitted without postprocess."
  echo "  task_jobs=$(join_by_colon "${JOB_IDS[@]}")"
  exit 0
fi

dependency="afterok:$(join_by_colon "${JOB_IDS[@]}")"
echo
echo "=== submitting official GeLLMO postprocess after $dependency ==="
SBATCH_ARGS=(
  --account="$POST_ACCOUNT"
  --job-name="$POST_JOB_NAME"
  --time="$POST_TIME"
  --mem="$POST_MEM"
  --cpus-per-task="$POST_CPUS"
  --dependency="$dependency"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL,SUCC_GELLMO_TASKS="$TASKS",SUCC_GELLMO_SETTINGS="$SETTINGS",SUCC_GELLMO_OUTPUT_DIR="$OUTPUT_DIR"
)
if [[ -n "$POST_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$POST_PARTITION")
fi
post_output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_gellmo_official_postprocess.sh'")"
echo "$post_output"
post_job_id="$(echo "$post_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$post_job_id" ]]; then
  echo "ERROR: failed to submit official GeLLMO postprocess." >&2
  exit 1
fi

echo
echo "Official GeLLMO MuMO suite submitted."
echo "  task_jobs=$(join_by_colon "${JOB_IDS[@]}")"
echo "  postprocess_job=$post_job_id"
echo "  output_dir=$OUTPUT_DIR"
