#!/usr/bin/env bash
# Submit SketchMol Understanding-Condition server smoke checks to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TIME="${SUCC_SLURM_TIME:-00:15:00}"
MEM="${SUCC_SLURM_MEM:-8G}"
CPUS="${SUCC_SLURM_CPUS:-1}"
JOB_NAME="${SUCC_SLURM_JOB_NAME:-succ-smoke}"
LOG_DIR="${SUCC_LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

export SUCC_TORCH_PYTHON="${SUCC_TORCH_PYTHON:-/home/bdong/scratch/venvs/phystabmol/bin/python}"

echo "Submitting SketchMol Understanding-Condition smoke:"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  torch_python=$SUCC_TORCH_PYTHON"
echo "  log_dir=$PROJECT_DIR/$LOG_DIR"

sbatch \
  --account="$ACCOUNT" \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$PROJECT_DIR/scripts/run_server_smoke.sh'"
