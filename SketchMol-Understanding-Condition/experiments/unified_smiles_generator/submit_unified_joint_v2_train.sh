#!/usr/bin/env bash
# Submit Unified Joint v2 U1/U2 training to Slurm.

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
ACCOUNT="${SUCC_UNIFIED_JOINT_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_JOINT_SLURM_TIME:-12:00:00}"
MEM="${SUCC_UNIFIED_JOINT_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_JOINT_SLURM_CPUS:-8}"
GPU_PROFILE="${SUCC_UNIFIED_JOINT_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_full}}"
PARTITION="${SUCC_UNIFIED_JOINT_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
JOB_NAME="${SUCC_UNIFIED_JOINT_JOB_NAME:-succ-unified-joint-v2-${STAGE}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_UNIFIED_JOINT_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_UNIFIED_JOINT_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"
SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --export=ALL,SUCC_UNIFIED_JOINT_STAGE="$STAGE"
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  output=""
  if output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_unified_joint_v2_train.sh'")"; then
    echo "$output"
    job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    [[ -n "$job_id" ]] && break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit Unified Joint v2 stage=$STAGE" >&2
  exit 1
fi
echo "unified_joint_v2_stage=$STAGE job_id=$job_id"
