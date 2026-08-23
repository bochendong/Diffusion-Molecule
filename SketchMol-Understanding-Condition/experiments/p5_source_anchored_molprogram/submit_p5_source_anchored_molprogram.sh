#!/usr/bin/env bash
# Submit the fast single-seed P5 gate on one available full H100.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P5_SEED:-7}"
ACCOUNT="${P5_SLURM_ACCOUNT:-rrg-hup}"
GPU="${P5_SLURM_GPU:-h100:1}"
TIME="${P5_SLURM_TIME:-02:00:00}"
MEM="${P5_SLURM_MEM:-64G}"
CPUS="${P5_SLURM_CPUS:-8}"
LOG_DIR="${P5_LOG_DIR:-$PROJECT_DIR/logs/p5_source_anchored_molprogram_v1}"
MAIL_USER="${P5_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="p5-source-copy-s${SEED}" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p5-source-copy-s${SEED}-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p5_source_anchored_molprogram.sh'")"
job_id="${submission%%;*}"
echo "P5 Source-Anchored MolProgram submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  time=$TIME"
echo "  log=$LOG_DIR/p5-source-copy-s${SEED}-${job_id}.log"
