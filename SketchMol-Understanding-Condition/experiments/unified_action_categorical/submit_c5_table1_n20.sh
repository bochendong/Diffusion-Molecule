#!/usr/bin/env bash
# Submit C5 graph-policy GRPO the B41 way: def-hup-ab, smallest MIG, 45 minutes.
# Do not use rrg-hup or a 4-hour wallclock; that is why 19982954 sat in Priority.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_C5_TRAIN_DEVICE="${SUCC_C5_TRAIN_DEVICE:-auto}"
export SUCC_DEVICE="${SUCC_DEVICE:-cpu}"
export SUCC_SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
ACCOUNT="${SUCC_C5_ACCOUNT:-def-hup-ab}"
TIME="${SUCC_C5_TIME:-00:45:00}"
MEM="${SUCC_C5_MEM:-20G}"
CPUS="${SUCC_C5_CPUS:-4}"
JOB_NAME="${SUCC_C5_JOB_NAME:-succ-c5-table1-n20}"
MAIL_USER="${SUCC_C5_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/joint_graph_fragment_categorical_c5}"

if [[ -n "${SUCC_C5_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_C5_GPUS")
else
  GPU_CANDIDATES=(
    "nvidia_h100_80gb_hbm3_1g.10gb:1"
    "nvidia_h100_80gb_hbm3_2g.20gb:1"
  )
fi

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  --parsable
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --mail-user="$MAIL_USER"
  --mail-type=END,FAIL
  --export=ALL
)

job_id=""
used_gpu=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gres=gpu:$GPU_REQUEST"
  if output="$(sbatch "${SBATCH_ARGS[@]}" --gres="gpu:$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_c5_table1_n20.sh'")"; then
    job_id="${output%%;*}"
    used_gpu="$GPU_REQUEST"
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit C5 Table1 n=20" >&2
  exit 1
fi

echo "c5_table1_n20_job=$job_id"
echo "gpu=$used_gpu"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
echo "summary=$PROJECT_DIR/outputs/joint_graph_fragment_categorical_c5/summary.json"
