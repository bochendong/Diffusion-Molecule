#!/bin/bash
# Submit Phase 1 oracle latent-conditioned SMILES diffusion.
#
# Usage from SketchImageJEPA:
#   SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" \
#   SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python \
#   SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv \
#   bash scripts/submit_oracle_latent_diffusion.sh

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

if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=(
    "/scratch/bdong/venvs/phystabmol/bin/python"
    "/home/bdong/scratch/venvs/phystabmol/bin/python"
    "/scratch/bdong/venvs/sketchimage-rdkit/bin/python"
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
fi
export SKETCHIMAGE_PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-$(command -v python3)}"

if ! "$SKETCHIMAGE_PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
PY
then
  cat <<EOF >&2
ERROR: $SKETCHIMAGE_PYTHON_BIN cannot import torch.

Find the torch venv first:
  SKETCHIMAGE_MODULES="gcc rdkit/2025.09.4" bash scripts/find_torch_python.sh

Then rerun with:
  SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python bash scripts/submit_oracle_latent_diffusion.sh
EOF
  exit 2
fi

if [[ -z "${SKETCHIMAGE_MOLECULE_CSV:-}" ]]; then
  echo "No molecule CSV provided; defaulting to data/example_molecules.csv for a tiny smoke job."
  export SKETCHIMAGE_MOLECULE_CSV="data/example_molecules.csv"
fi

if [[ ! -f "$SKETCHIMAGE_MOLECULE_CSV" ]]; then
  echo "ERROR: SKETCHIMAGE_MOLECULE_CSV does not exist: $SKETCHIMAGE_MOLECULE_CSV" >&2
  exit 2
fi

mkdir -p outputs/logs

export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase1_oracle_latent_diffusion_seed${SKETCHIMAGE_SEED:-7}}"
export SKETCHIMAGE_MOLECULE_LIMIT="${SKETCHIMAGE_MOLECULE_LIMIT:-50000}"
export SKETCHIMAGE_ORACLE_EPOCHS="${SKETCHIMAGE_ORACLE_EPOCHS:-20}"
export SKETCHIMAGE_ORACLE_BATCH_SIZE="${SKETCHIMAGE_ORACLE_BATCH_SIZE:-128}"
export SKETCHIMAGE_ORACLE_HIDDEN_DIM="${SKETCHIMAGE_ORACLE_HIDDEN_DIM:-256}"
export SKETCHIMAGE_ORACLE_OBJECTIVE="${SKETCHIMAGE_ORACLE_OBJECTIVE:-autoregressive}"
export SKETCHIMAGE_ORACLE_SAMPLE_STEPS="${SKETCHIMAGE_ORACLE_SAMPLE_STEPS:-16}"
export SKETCHIMAGE_ORACLE_SAMPLES_PER_CONDITION="${SKETCHIMAGE_ORACLE_SAMPLES_PER_CONDITION:-8}"

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

SLURM_TIME="${SKETCHIMAGE_SLURM_TIME:-04:00:00}"
SLURM_MEM="${SKETCHIMAGE_SLURM_MEM:-32G}"
SLURM_CPUS="${SKETCHIMAGE_SLURM_CPUS:-4}"

echo "Submitting Phase 1 oracle latent diffusion:"
echo "  run_name=$SKETCHIMAGE_RUN_NAME"
echo "  molecule_csv=$SKETCHIMAGE_MOLECULE_CSV"
echo "  molecule_limit=$SKETCHIMAGE_MOLECULE_LIMIT"
echo "  python=$SKETCHIMAGE_PYTHON_BIN"
echo "  epochs=$SKETCHIMAGE_ORACLE_EPOCHS"
echo "  batch_size=$SKETCHIMAGE_ORACLE_BATCH_SIZE"
echo "  hidden_dim=$SKETCHIMAGE_ORACLE_HIDDEN_DIM"
echo "  objective=$SKETCHIMAGE_ORACLE_OBJECTIVE"
echo "  sample_steps=$SKETCHIMAGE_ORACLE_SAMPLE_STEPS"
echo "  samples_per_condition=$SKETCHIMAGE_ORACLE_SAMPLES_PER_CONDITION"
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
    --output="./outputs/logs/sketchimage-phase1-%j.log" \
    --error="./outputs/logs/sketchimage-phase1-%j.log" \
    scripts/run_oracle_latent_diffusion.slurm.sh; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: none of the GPU candidates were accepted by Slurm." >&2
  echo "Try SKETCHIMAGE_SLURM_GPUS=<available_gpu_name>:1 bash scripts/submit_oracle_latent_diffusion.sh" >&2
  exit 2
fi
