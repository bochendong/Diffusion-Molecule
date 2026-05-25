#!/bin/bash
# Submit benchmark audit to a CPU Slurm node.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

mkdir -p outputs/logs

if [[ $# -gt 0 ]]; then
  export SKETCHIMAGE_RUN_DIR="$1"
fi

if [[ -z "${SKETCHIMAGE_RUN_DIR:-}" && ( -z "${SKETCHIMAGE_TRAIN_CSV:-}" || -z "${SKETCHIMAGE_EVAL_CSV:-}" ) ]]; then
  echo "ERROR: provide a run directory, or set SKETCHIMAGE_TRAIN_CSV and SKETCHIMAGE_EVAL_CSV." >&2
  exit 2
fi

export SKETCHIMAGE_MODULES="${SKETCHIMAGE_MODULES:-gcc rdkit/2025.09.4}"

SLURM_TIME="${SKETCHIMAGE_CPU_SLURM_TIME:-02:00:00}"
SLURM_MEM="${SKETCHIMAGE_CPU_SLURM_MEM:-32G}"
SLURM_CPUS="${SKETCHIMAGE_CPU_SLURM_CPUS:-8}"

echo "Submitting SketchImage-JEPA audit CPU job:"
echo "  run_dir=${SKETCHIMAGE_RUN_DIR:-<not provided>}"
echo "  train_csv=${SKETCHIMAGE_TRAIN_CSV:-<not provided>}"
echo "  eval_csv=${SKETCHIMAGE_EVAL_CSV:-<not provided>}"
echo "  python=${SKETCHIMAGE_PYTHON_BIN:-<auto>}"
echo "  modules=$SKETCHIMAGE_MODULES"
echo "  slurm_time=$SLURM_TIME"
echo "  slurm_mem=$SLURM_MEM"
echo "  slurm_cpus=$SLURM_CPUS"

sbatch \
  --export=ALL \
  --time="$SLURM_TIME" \
  --mem="$SLURM_MEM" \
  --cpus-per-task="$SLURM_CPUS" \
  --output="./outputs/logs/sketchimage-cpu-%j.log" \
  --error="./outputs/logs/sketchimage-cpu-%j.log" \
  scripts/run_audit_benchmark.slurm.sh
