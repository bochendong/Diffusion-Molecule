#!/usr/bin/env bash
# Submit the full Unified Joint v2 evaluation matrix to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
STAGE="${SUCC_UNIFIED_JOINT_STAGE:-u2}"
ACCOUNT="${SUCC_UNIFIED_JOINT_EVAL_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_JOINT_EVAL_SLURM_TIME:-24:00:00}"
MEM="${SUCC_UNIFIED_JOINT_EVAL_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_JOINT_EVAL_SLURM_CPUS:-8}"
GPU="${SUCC_UNIFIED_JOINT_EVAL_SLURM_GPUS:-h100:1}"
PARTITION="${SUCC_UNIFIED_JOINT_EVAL_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
JOB_NAME="${SUCC_UNIFIED_JOINT_EVAL_JOB_NAME:-succ-unified-joint-v2-${STAGE}-eval}"

mkdir -p "$LOG_DIR"
SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --gpus="$GPU"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --export=ALL,SUCC_UNIFIED_JOINT_STAGE="$STAGE"
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$SCRIPT_DIR/run_unified_joint_v2_eval_suite.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit Unified Joint v2 evaluation" >&2
  exit 1
fi
echo "unified_joint_v2_eval_stage=$STAGE job_id=$job_id"
