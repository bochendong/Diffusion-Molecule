#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 R1_JOB_ID" >&2
  exit 2
fi
R1_JOB_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P811_SEED:-7}"
LOG_DIR="${P811_LOG_DIR:-$PROJECT_DIR/logs/p8_1_1_short_transaction}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P811_SLURM_ACCOUNT:-rrg-hup}" \
  --dependency="afterok:${R1_JOB_ID}" --job-name="p811-r2-s${SEED}" \
  --time="${P811_R2_SLURM_TIME:-00:05:00}" --mem="${P811_SLURM_MEM:-48G}" \
  --cpus-per-task="${P811_SLURM_CPUS:-8}" --gpus="${P811_SLURM_GPU:-h100:1}" \
  --mail-user="${P811_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p811-r2-s${SEED}-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p8_1_1_r2_temperature.sh'")"
job_id="${submission%%;*}"
echo "P8.1.1-R2 temperature intervention submitted"
echo "  job_id=$job_id"
echo "  dependency=afterok:$R1_JOB_ID"
echo "  log=$LOG_DIR/p811-r2-s${SEED}-${job_id}.log"

