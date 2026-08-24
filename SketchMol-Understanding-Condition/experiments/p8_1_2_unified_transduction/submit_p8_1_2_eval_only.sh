#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P812_SEED:-7}"
ROUND="${P812_EVAL_ROUND:?set P812_EVAL_ROUND to r1 or r2}"
OUTPUT_ROOT="${P812_OUTPUT_ROOT:?set P812_OUTPUT_ROOT to the completed training root}"
LOG_DIR="${P812_EVAL_LOG_DIR:-$PROJECT_DIR/logs/p8_1_2_unified_transduction_eval_only_v1}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P812_SLURM_ACCOUNT:-rrg-hup}" \
  --job-name="p812-${ROUND}-eval-s${SEED}" --time="${P812_SLURM_TIME:-00:06:00}" \
  --mem="${P812_SLURM_MEM:-48G}" --cpus-per-task="${P812_SLURM_CPUS:-8}" \
  --gpus="${P812_SLURM_GPU:-h100:1}" --output="$LOG_DIR/p812-${ROUND}-eval-s${SEED}-%j.log" \
  --export=ALL --wrap="P812_OUTPUT_ROOT='$OUTPUT_ROOT' bash '$SCRIPT_DIR/run_p8_1_2_eval_only.sh'")"
job_id="${submission%%;*}"
echo "P8.1.2 ${ROUND} eval-only continuation submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/p812-${ROUND}-eval-s${SEED}-${job_id}.log"
