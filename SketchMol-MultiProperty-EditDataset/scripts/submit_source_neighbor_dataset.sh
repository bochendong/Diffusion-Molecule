#!/usr/bin/env bash
# Submit the source-neighbor dataset build to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

ACCOUNT="${SMMED_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TIME="${SMMED_SLURM_TIME:-04:00:00}"
MEM="${SMMED_SLURM_MEM:-16G}"
CPUS="${SMMED_SLURM_CPUS:-1}"
JOB_NAME="${SMMED_SLURM_JOB_NAME:-smmed-src-neighbor}"
LOG_DIR="${SMMED_LOG_DIR:-$PROJECT_DIR/logs}"

mkdir -p "$LOG_DIR"

echo "Submitting source-neighbor multi-property dataset build"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  log_dir=$LOG_DIR"

sbatch \
  --account="$ACCOUNT" \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$PROJECT_DIR/scripts/run_build_source_neighbor_dataset.sh'"
