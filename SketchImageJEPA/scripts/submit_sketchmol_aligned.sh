#!/bin/bash
# Submit the SketchMol-aligned SketchImage-JEPA experiment from a Slurm login node.
#
# Usage from SketchImageJEPA:
#   SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv bash scripts/submit_sketchmol_aligned.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    export SKETCHIMAGE_PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    export SKETCHIMAGE_PYTHON_BIN="$(command -v python3)"
  fi
fi

if [[ -z "${SKETCHIMAGE_MOLECULE_CSV:-}" && -z "${SKETCHIMAGE_DATASET_CSV:-}" ]]; then
  echo "No molecule/task CSV provided; defaulting to data/example_molecules.csv for a small smoke submission."
  export SKETCHIMAGE_MOLECULE_CSV="data/example_molecules.csv"
fi

check_csv_exists() {
  local label="$1"
  local path="$2"
  if [[ -n "$path" && ! -f "$path" ]]; then
    cat <<EOF >&2
ERROR: $label does not exist: $path

Use a real CSV path. From this project you can test with:
  SKETCHIMAGE_MOLECULE_CSV=data/example_molecules.csv bash scripts/submit_sketchmol_aligned.sh

Or find candidate CSVs with:
  find /scratch/bdong/projects/Diffusion-Molecule -name '*.csv' | head -50
EOF
    exit 2
  fi
}

check_csv_exists "SKETCHIMAGE_MOLECULE_CSV" "${SKETCHIMAGE_MOLECULE_CSV:-}"
check_csv_exists "SKETCHIMAGE_DATASET_CSV" "${SKETCHIMAGE_DATASET_CSV:-}"
check_csv_exists "SKETCHIMAGE_TRAIN_CSV" "${SKETCHIMAGE_TRAIN_CSV:-}"
check_csv_exists "SKETCHIMAGE_EVAL_CSV" "${SKETCHIMAGE_EVAL_CSV:-}"

mkdir -p outputs/logs

export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-sketchimage_aligned_${SKETCHIMAGE_MOLECULE_LIMIT:-10000}_$(date +%Y%m%d_%H%M%S)}"
export SKETCHIMAGE_MOLECULE_LIMIT="${SKETCHIMAGE_MOLECULE_LIMIT:-10000}"
export SKETCHIMAGE_MAX_TASKS="${SKETCHIMAGE_MAX_TASKS:-5000}"

SLURM_TIME="${SKETCHIMAGE_SLURM_TIME:-02:00:00}"
SLURM_MEM="${SKETCHIMAGE_SLURM_MEM:-32G}"
SLURM_CPUS="${SKETCHIMAGE_SLURM_CPUS:-8}"

echo "Submitting SketchImage-JEPA SketchMol-aligned run:"
echo "  run_name=$SKETCHIMAGE_RUN_NAME"
echo "  molecule_csv=${SKETCHIMAGE_MOLECULE_CSV:-<not provided>}"
echo "  dataset_csv=${SKETCHIMAGE_DATASET_CSV:-<not provided>}"
echo "  molecule_limit=$SKETCHIMAGE_MOLECULE_LIMIT"
echo "  max_tasks=$SKETCHIMAGE_MAX_TASKS"
echo "  python=$SKETCHIMAGE_PYTHON_BIN"
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
  scripts/run_sketchmol_aligned.slurm.sh
