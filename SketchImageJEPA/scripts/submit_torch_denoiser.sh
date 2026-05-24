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

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/sketchimage-rdkit/bin/python"
if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=(
    "$DEFAULT_SERVER_PYTHON"
    "/scratch/bdong/venvs/phystabmol/bin/python"
    "/home/bdong/scratch/venvs/phystabmol/bin/python"
    "$(command -v python3 2>/dev/null || true)"
  )
  for python_candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$python_candidate" && -x "$python_candidate" ]] && "$python_candidate" - <<'PY' >/dev/null 2>&1
import torch
PY
    then
      export SKETCHIMAGE_PYTHON_BIN="$python_candidate"
      break
    fi
  done
  if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
    if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
      export SKETCHIMAGE_PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
    else
      export SKETCHIMAGE_PYTHON_BIN="$(command -v python3)"
    fi
  fi
fi

if [[ "${SKETCHIMAGE_TORCH_PREFLIGHT:-1}" == "1" ]]; then
  if ! "$SKETCHIMAGE_PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
PY
  then
    cat <<EOF >&2
ERROR: $SKETCHIMAGE_PYTHON_BIN cannot import torch.

Set SKETCHIMAGE_PYTHON_BIN to a Python with PyTorch, or install torch into the
current venv first:

  MODULE_RDKIT=rdkit/2025.09.4 \\
  VENV_DIR=/scratch/bdong/venvs/sketchimage-rdkit \\
  bash scripts/setup_torch_venv.sh

Or locate the existing torch venv first:

  bash scripts/find_torch_python.sh

If torch import only works after special modules are loaded inside Slurm, rerun
with SKETCHIMAGE_TORCH_PREFLIGHT=0 and set SKETCHIMAGE_MODULES accordingly.
EOF
    exit 2
  fi
fi

if [[ "${SKETCHIMAGE_RDKIT_PREFLIGHT:-1}" == "1" ]]; then
  if ! "$SKETCHIMAGE_PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import rdkit
PY
  then
    cat <<EOF >&2
ERROR: $SKETCHIMAGE_PYTHON_BIN cannot import rdkit.

This cluster exposes RDKit through modules. Try:

  SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \\
  SKETCHIMAGE_PYTHON_BIN=$SKETCHIMAGE_PYTHON_BIN \\
  bash scripts/submit_torch_denoiser.sh

You can inspect candidates with:

  SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" bash scripts/find_torch_python.sh

If you intentionally want fallback string metrics, rerun with
SKETCHIMAGE_RDKIT_PREFLIGHT=0.
EOF
    exit 2
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
export SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT:-1.0}"
export SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT:-1.0}"
export SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT:-8.0}"
export SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT="${SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT:-0.05}"
export SKETCHIMAGE_SOURCE_RERANK_WEIGHT="${SKETCHIMAGE_SOURCE_RERANK_WEIGHT:-0.35}"
export SKETCHIMAGE_PROPERTY_RERANK_WEIGHT="${SKETCHIMAGE_PROPERTY_RERANK_WEIGHT:-0.25}"
export SKETCHIMAGE_SCAFFOLD_RERANK_BONUS="${SKETCHIMAGE_SCAFFOLD_RERANK_BONUS:-0.15}"

GPU_PROFILE="${SKETCHIMAGE_GPU_PROFILE:-h100_10gb_mig}"
if [[ -n "${SKETCHIMAGE_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SKETCHIMAGE_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1" "h100_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1" "nvidia_h100_80gb_hbm3_3g.40gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100_80gb:1" "h100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

SLURM_TIME="${SKETCHIMAGE_SLURM_TIME:-08:00:00}"
SLURM_MEM="${SKETCHIMAGE_SLURM_MEM:-64G}"
SLURM_CPUS="${SKETCHIMAGE_SLURM_CPUS:-8}"

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
echo "  torch_direct_loss_weight=$SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT"
echo "  torch_cosine_loss_weight=$SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT"
echo "  torch_positive_loss_weight=$SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT"
echo "  de_novo_latent_rerank_weight=$SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT"
echo "  source_rerank_weight=$SKETCHIMAGE_SOURCE_RERANK_WEIGHT"
echo "  property_rerank_weight=$SKETCHIMAGE_PROPERTY_RERANK_WEIGHT"
echo "  scaffold_rerank_bonus=$SKETCHIMAGE_SCAFFOLD_RERANK_BONUS"
echo "  gpu_profile=$GPU_PROFILE"
echo "  slurm_gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  slurm_time=$SLURM_TIME"
echo "  slurm_mem=$SLURM_MEM"
echo "  slurm_cpus=$SLURM_CPUS"

SUBMITTED=0
for SLURM_GPUS in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$SLURM_GPUS"
  if sbatch \
    --export=ALL \
    --gpus="$SLURM_GPUS" \
    --time="$SLURM_TIME" \
    --mem="$SLURM_MEM" \
    --cpus-per-task="$SLURM_CPUS" \
    scripts/run_torch_denoiser.slurm.sh; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: none of the GPU candidates were accepted by Slurm." >&2
  echo "Try checking available names with:" >&2
  echo "  sinfo -o '%G %N' | sort -u" >&2
  echo "Then rerun with:" >&2
  echo "  SKETCHIMAGE_SLURM_GPUS=<available_gpu_name>:1 bash scripts/submit_torch_denoiser.sh" >&2
  exit 2
fi
