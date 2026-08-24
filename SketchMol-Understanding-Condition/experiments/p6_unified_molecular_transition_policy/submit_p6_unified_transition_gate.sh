#!/usr/bin/env bash
# Submit the bounded single-seed P6 gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P6_SEED:-7}"
ACCOUNT="${P6_SLURM_ACCOUNT:-rrg-hup}"
GPU="${P6_SLURM_GPU:-h100:1}"
TIME="${P6_SLURM_TIME:-03:00:00}"
MEM="${P6_SLURM_MEM:-64G}"
CPUS="${P6_SLURM_CPUS:-8}"
LOG_DIR="${P6_LOG_DIR:-$PROJECT_DIR/logs/p6_unified_transition_policy_v1}"
MAIL_USER="${P6_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable --account="$ACCOUNT" --job-name="p6-unified-s${SEED}" \
  --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS" --gpus="$GPU" \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p6-unified-s${SEED}-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p6_unified_transition_gate.sh'")"
job_id="${submission%%;*}"
echo "P6 unified transition gate submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  time=$TIME"
echo "  log=$LOG_DIR/p6-unified-s${SEED}-${job_id}.log"
