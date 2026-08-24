#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P812_SEED:-7}"
LOG_DIR="${P812_LOG_DIR:-$PROJECT_DIR/logs/p8_1_2_unified_transduction_v1}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P812_SLURM_ACCOUNT:-def-hup-ab}" \
  --job-name="p812-prep-s${SEED}" --time="${P812_SLURM_TIME:-00:05:00}" \
  --mem="${P812_SLURM_MEM:-32G}" --cpus-per-task="${P812_SLURM_CPUS:-8}" \
  --output="$LOG_DIR/p812-prep-s${SEED}-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p8_1_2_prepare_validation.sh'")"
job_id="${submission%%;*}"
echo "P8.1.2 validation R2 preparation submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/p812-prep-s${SEED}-${job_id}.log"
