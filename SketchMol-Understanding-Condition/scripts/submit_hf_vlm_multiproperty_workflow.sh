#!/usr/bin/env bash
# Submit the big-VLM multi-property understanding workflow to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TIME="${SUCC_SLURM_TIME:-24:00:00}"
MEM="${SUCC_SLURM_MEM:-80G}"
CPUS="${SUCC_SLURM_CPUS:-8}"
GPUS="${SUCC_SLURM_GPUS:-1}"
JOB_NAME="${SUCC_SLURM_JOB_NAME:-succ-hf-vlm}"
LOG_DIR="${SUCC_LOG_DIR:-logs}"
PARTITION="${SUCC_SLURM_PARTITION:-}"

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --gres="gpu:$GPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

echo "Submitting HF VLM multi-property workflow:"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  gpus=$GPUS"
echo "  log_dir=$PROJECT_DIR/$LOG_DIR"

sbatch "${SBATCH_ARGS[@]}" \
  --wrap="bash '$PROJECT_DIR/scripts/run_hf_vlm_multiproperty_workflow.sh'"
