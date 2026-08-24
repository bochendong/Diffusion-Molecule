#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P8_SEED:-7}"
LOG_DIR="${P8_LOG_DIR:-$PROJECT_DIR/logs/p8_validity_preserving_molprogram_v1}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P8_SLURM_ACCOUNT:-rrg-hup}" \
  --job-name="p8-validstop-s${SEED}" --time="${P8_SLURM_TIME:-00:10:00}" \
  --mem="${P8_SLURM_MEM:-64G}" --cpus-per-task="${P8_SLURM_CPUS:-8}" \
  --gpus="${P8_SLURM_GPU:-h100:1}" --mail-user="${P8_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" \
  --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/p8-validstop-s${SEED}-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_p8_validity_gate.sh'")"
job_id="${submission%%;*}"
echo "P8 validity-preserving gate submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/p8-validstop-s${SEED}-${job_id}.log"
