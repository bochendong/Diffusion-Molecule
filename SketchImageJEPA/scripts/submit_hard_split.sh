#!/bin/bash
# Submit hard split construction to a CPU Slurm node.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

mkdir -p outputs/logs

export SKETCHIMAGE_MODULES="${SKETCHIMAGE_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHIMAGE_MOLECULE_LIMIT="${SKETCHIMAGE_MOLECULE_LIMIT:-50000}"
export SKETCHIMAGE_MAX_TASKS="${SKETCHIMAGE_MAX_TASKS:-10000}"
export SKETCHIMAGE_HARD_SPLIT_NAME="${SKETCHIMAGE_HARD_SPLIT_NAME:-sketchmol_hard_seed${SKETCHIMAGE_SEED:-7}}"

SLURM_TIME="${SKETCHIMAGE_CPU_SLURM_TIME:-04:00:00}"
SLURM_MEM="${SKETCHIMAGE_CPU_SLURM_MEM:-32G}"
SLURM_CPUS="${SKETCHIMAGE_CPU_SLURM_CPUS:-8}"

echo "Submitting SketchImage-JEPA hard split CPU job:"
echo "  split_name=$SKETCHIMAGE_HARD_SPLIT_NAME"
echo "  molecule_csv=${SKETCHIMAGE_MOLECULE_CSV:-<not provided>}"
echo "  task_csv=${SKETCHIMAGE_TASK_CSV:-outputs/tasks/${SKETCHIMAGE_HARD_SPLIT_NAME}_tasks.csv}"
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
  scripts/run_hard_split.slurm.sh
