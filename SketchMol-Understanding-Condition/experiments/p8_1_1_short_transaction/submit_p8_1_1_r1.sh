#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P811_SEED:-7}"
LOG_DIR="${P811_LOG_DIR:-$PROJECT_DIR/logs/p8_1_1_short_transaction}"
GPU="${P811_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P811_SLURM_ACCOUNT:-rrg-hup}" \
  --job-name="p811-r1-s${SEED}" --time="${P811_SLURM_TIME:-00:30:00}" \
  --mem="${P811_SLURM_MEM:-48G}" --cpus-per-task="${P811_SLURM_CPUS:-8}" \
  --gpus="$GPU" --mail-user="${P811_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" \
  --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/p811-r1-s${SEED}-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_p8_1_1_r1.sh'")"
job_id="${submission%%;*}"
echo "P8.1.1-R1 submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  log=$LOG_DIR/p811-r1-s${SEED}-${job_id}.log"

