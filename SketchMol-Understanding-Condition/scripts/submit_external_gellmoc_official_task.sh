#!/usr/bin/env bash
# Submit one official GeLLMO-C C-MuMO inference task on Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

TASK="${SUCC_GELLMOC_TASK:-BPQ}"
SETTING="${SUCC_GELLMOC_SETTING:-seen}"
JOB_NAME="${SUCC_GELLMOC_SLURM_JOB_NAME:-succ-gellmoc-official-${TASK//+/-}-${SETTING}}"
ACCOUNT="${SUCC_GELLMOC_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_GELLMOC_SLURM_TIME:-12:00:00}"
CPUS="${SUCC_GELLMOC_SLURM_CPUS:-8}"
MEM="${SUCC_GELLMOC_SLURM_MEM:-96G}"
GPUS="${SUCC_GELLMOC_SLURM_GPUS:-a100:1}"
PARTITION="${SUCC_GELLMOC_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
DEPENDENCY="${SUCC_GELLMOC_SLURM_DEPENDENCY:-${SUCC_SLURM_DEPENDENCY:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"

mkdir -p "$LOG_DIR"

echo "Submitting official GeLLMO-C task"
echo "  task=$TASK"
echo "  setting=$SETTING"
echo "  job_name=$JOB_NAME"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  gpus=${GPUS:-none}"
echo "  dependency=${DEPENDENCY:-none}"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL,SUCC_GELLMOC_TASK="$TASK",SUCC_GELLMOC_SETTING="$SETTING"
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi
if [[ -n "$GPUS" && "$GPUS" != "none" && "$GPUS" != "0" ]]; then
  SBATCH_ARGS+=(--gpus="$GPUS")
fi

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_gellmoc_official_task.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit official GeLLMO-C task." >&2
  exit 1
fi

echo "external_gellmoc_official_task_job=$job_id"
