#!/bin/bash
# Submit the GPU-capable PyTorch latent denoising SketchImage-JEPA run.
#
# Usage from SketchImageJEPA:
#   SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv bash scripts/submit_torch_denoiser.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/sketchimage-rdkit/bin/python"
if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    export SKETCHIMAGE_PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    export SKETCHIMAGE_PYTHON_BIN="$(command -v python3)"
  fi
fi

if [[ -z "${SKETCHIMAGE_MOLECULE_CSV:-}" && -z "${SKETCHIMAGE_DATASET_CSV:-}" ]]; then
  echo "No molecule/task CSV provided; defaulting to data/example_molecules.csv for a small GPU smoke submission."
  export SKETCHIMAGE_MOLECULE_CSV="data/example_molecules.csv"
fi

check_csv_exists() {
  local label="$1"
  local path="$2"
  if [[ -n "$path" && ! -f "$path" ]]; then
    cat <<EOF >&2
ERROR: $label does not exist: $path

Use a real CSV path. From this project you can test with:
  SKETCHIMAGE_MOLECULE_CSV=data/example_molecules.csv bash scripts/submit_torch_denoiser.sh
EOF
    exit 2
  fi
}

check_csv_exists "SKETCHIMAGE_MOLECULE_CSV" "${SKETCHIMAGE_MOLECULE_CSV:-}"
check_csv_exists "SKETCHIMAGE_DATASET_CSV" "${SKETCHIMAGE_DATASET_CSV:-}"
check_csv_exists "SKETCHIMAGE_TRAIN_CSV" "${SKETCHIMAGE_TRAIN_CSV:-}"
check_csv_exists "SKETCHIMAGE_EVAL_CSV" "${SKETCHIMAGE_EVAL_CSV:-}"

export SKETCHIMAGE_BACKEND="${SKETCHIMAGE_BACKEND:-torch_denoiser}"
export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-sketchimage_torch_${SKETCHIMAGE_MOLECULE_LIMIT:-10000}_$(date +%Y%m%d_%H%M%S)}"
export SKETCHIMAGE_MOLECULE_LIMIT="${SKETCHIMAGE_MOLECULE_LIMIT:-10000}"
export SKETCHIMAGE_MAX_TASKS="${SKETCHIMAGE_MAX_TASKS:-5000}"
export SKETCHIMAGE_TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-25}"
export SKETCHIMAGE_TORCH_BATCH_SIZE="${SKETCHIMAGE_TORCH_BATCH_SIZE:-128}"
export SKETCHIMAGE_TORCH_HIDDEN_DIM="${SKETCHIMAGE_TORCH_HIDDEN_DIM:-1024}"
export SKETCHIMAGE_TORCH_DIFFUSION_STEPS="${SKETCHIMAGE_TORCH_DIFFUSION_STEPS:-16}"

SLURM_TIME="${SKETCHIMAGE_SLURM_TIME:-08:00:00}"
SLURM_MEM="${SKETCHIMAGE_SLURM_MEM:-64G}"
SLURM_CPUS="${SKETCHIMAGE_SLURM_CPUS:-8}"
SLURM_GPUS="${SKETCHIMAGE_SLURM_GPUS:-1}"

echo "Submitting SketchImage-JEPA torch denoiser run:"
echo "  run_name=$SKETCHIMAGE_RUN_NAME"
echo "  molecule_csv=${SKETCHIMAGE_MOLECULE_CSV:-<not provided>}"
echo "  dataset_csv=${SKETCHIMAGE_DATASET_CSV:-<not provided>}"
echo "  molecule_limit=$SKETCHIMAGE_MOLECULE_LIMIT"
echo "  max_tasks=$SKETCHIMAGE_MAX_TASKS"
echo "  python=$SKETCHIMAGE_PYTHON_BIN"
echo "  torch_epochs=$SKETCHIMAGE_TORCH_EPOCHS"
echo "  torch_batch_size=$SKETCHIMAGE_TORCH_BATCH_SIZE"
echo "  torch_hidden_dim=$SKETCHIMAGE_TORCH_HIDDEN_DIM"
echo "  slurm_time=$SLURM_TIME"
echo "  slurm_mem=$SLURM_MEM"
echo "  slurm_cpus=$SLURM_CPUS"
echo "  slurm_gpus=$SLURM_GPUS"

sbatch \
  --export=ALL \
  --gres="gpu:${SLURM_GPUS}" \
  --time="$SLURM_TIME" \
  --mem="$SLURM_MEM" \
  --cpus-per-task="$SLURM_CPUS" \
  scripts/run_torch_denoiser.slurm.sh
