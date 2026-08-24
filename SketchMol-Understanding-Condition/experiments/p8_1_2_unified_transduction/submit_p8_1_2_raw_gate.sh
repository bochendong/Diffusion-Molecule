#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P812_SEED:-7}"
LOG_DIR="${P812_LOG_DIR:-$PROJECT_DIR/logs/p8_1_2_unified_transduction_raw_v1}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P812_SLURM_ACCOUNT:-def-hup-ab}" \
  --job-name="p812-raw-s${SEED}" --time="${P812_SLURM_TIME:-00:30:00}" \
  --mem="${P812_SLURM_MEM:-64G}" --cpus-per-task="${P812_SLURM_CPUS:-8}" \
  --gpus="${P812_SLURM_GPU:-h100:1}" \
  --mail-user="${P812_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p812-raw-s${SEED}-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_p8_1_2_raw_gate.sh'")"
job_id="${submission%%;*}"
echo "P8.1.2 raw unified gate submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/p812-raw-s${SEED}-${job_id}.log"
