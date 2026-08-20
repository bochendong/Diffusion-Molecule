#!/usr/bin/env bash
# Submit D3: train energy into B41 event kernel, Table1 n=20.
# Budget: MCS ~2 min (D2 1500 pairs was 106s) + supervised 3 epochs +
# GRPO 80 conditions x 8 x 2 + B41 eval. D1 eval was 56 min at 3.4 s/cond;
# this eval is plain B41 (~2-3 s/cond) ≈ 35-50 min. Request 2.5 h buffer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_D3_DEVICE="${SUCC_D3_DEVICE:-auto}"
export SUCC_DEVICE="${SUCC_DEVICE:-cpu}"
export SUCC_SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
ACCOUNT="${SUCC_D3_ACCOUNT:-def-hup-ab}"
TIME="${SUCC_D3_TIME:-03:30:00}"
MEM="${SUCC_D3_MEM:-20G}"
CPUS="${SUCC_D3_CPUS:-4}"
JOB_NAME="${SUCC_D3_JOB_NAME:-succ-d3-event-kernel-energy}"
MAIL_USER="${SUCC_D0_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/d3_event_kernel_energy}"

if [[ -n "${SUCC_D3_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_D3_GPUS")
else
  GPU_CANDIDATES=(
    "nvidia_h100_80gb_hbm3_2g.20gb:1"
    "nvidia_h100_80gb_hbm3_3g.40gb:1"
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
  if output="$(sbatch "${SBATCH_ARGS[@]}" --gres="gpu:$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_d3_event_kernel_energy.sh'")"; then
    job_id="${output%%;*}"
    used_gpu="$GPU_REQUEST"
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit D3 event-kernel energy Table1 n=20" >&2
  exit 1
fi

echo "d3_event_kernel_energy_job=$job_id"
echo "gpu=$used_gpu"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
echo "summary=${SUCC_D3_OUTPUT_DIR:-$PROJECT_DIR/outputs/d3_event_kernel_energy_table1_n20}/summary.json"
