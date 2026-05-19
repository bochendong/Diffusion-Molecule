#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA="${MOLPILOT_DATA:-../PhysTabMol/data/molecules.csv}"
LIMIT="${MOLPILOT_LIMIT:-10000}"
EVAL_LIMIT="${MOLPILOT_EVAL_LIMIT:-1000}"
CODEC="${MOLPILOT_CODEC:-sequence}"
RUN_NAME="${MOLPILOT_RUN_NAME:-molpilot_${CODEC}_${LIMIT}_$(date +%Y%m%d_%H%M%S)}"
GPU_PROFILE="${MOLPILOT_GPU_PROFILE:-h100_40gb_mig}"
SLURM_MEM_PER_CPU="${MOLPILOT_SLURM_MEM_PER_CPU:-4096M}"

if [[ -n "${MOLPILOT_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$MOLPILOT_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("h100_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100_80gb:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: molecule CSV not found: $DATA"
  echo "Expected the ChEMBL file created by PhysTabMol, usually:"
  echo "  ../PhysTabMol/data/molecules.csv"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch was not found. Run this on a Slurm login node."
  exit 2
fi

echo "Submitting MolPilot staged ChEMBL run:"
echo "  data=$DATA"
echo "  limit=$LIMIT"
echo "  eval_limit=$EVAL_LIMIT"
echo "  codec=$CODEC"
echo "  gpu_profile=$GPU_PROFILE"
echo "  slurm_gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  slurm_mem_per_cpu=$SLURM_MEM_PER_CPU"
echo "  run_name=$RUN_NAME"

export MOLPILOT_DATA="$DATA"
export MOLPILOT_LIMIT="$LIMIT"
export MOLPILOT_EVAL_LIMIT="$EVAL_LIMIT"
export MOLPILOT_CODEC="$CODEC"
export MOLPILOT_RUN_NAME="$RUN_NAME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  export MOLPILOT_EMBEDDING_DIM="${MOLPILOT_EMBEDDING_DIM:-128}"
  export MOLPILOT_MAX_LENGTH="${MOLPILOT_MAX_LENGTH:-96}"
  export MOLPILOT_AE_HIDDEN_DIM="${MOLPILOT_AE_HIDDEN_DIM:-384}"
  export MOLPILOT_AE_LAYERS="${MOLPILOT_AE_LAYERS:-2}"
  export MOLPILOT_AE_BATCH_SIZE="${MOLPILOT_AE_BATCH_SIZE:-128}"
  export MOLPILOT_ALIGN_HIDDEN_DIM="${MOLPILOT_ALIGN_HIDDEN_DIM:-512}"
  export MOLPILOT_ALIGN_LAYERS="${MOLPILOT_ALIGN_LAYERS:-3}"
  export MOLPILOT_ALIGN_BATCH_SIZE="${MOLPILOT_ALIGN_BATCH_SIZE:-512}"
  export MOLPILOT_DIFFUSION_HIDDEN_DIM="${MOLPILOT_DIFFUSION_HIDDEN_DIM:-512}"
  export MOLPILOT_DIFFUSION_LAYERS="${MOLPILOT_DIFFUSION_LAYERS:-4}"
  export MOLPILOT_DIFFUSION_BATCH_SIZE="${MOLPILOT_DIFFUSION_BATCH_SIZE:-512}"
  export MOLPILOT_TIMESTEPS="${MOLPILOT_TIMESTEPS:-50}"
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  export MOLPILOT_EMBEDDING_DIM="${MOLPILOT_EMBEDDING_DIM:-192}"
  export MOLPILOT_MAX_LENGTH="${MOLPILOT_MAX_LENGTH:-128}"
  export MOLPILOT_AE_HIDDEN_DIM="${MOLPILOT_AE_HIDDEN_DIM:-768}"
  export MOLPILOT_AE_LAYERS="${MOLPILOT_AE_LAYERS:-4}"
  export MOLPILOT_AE_BATCH_SIZE="${MOLPILOT_AE_BATCH_SIZE:-512}"
  export MOLPILOT_ALIGN_HIDDEN_DIM="${MOLPILOT_ALIGN_HIDDEN_DIM:-768}"
  export MOLPILOT_ALIGN_LAYERS="${MOLPILOT_ALIGN_LAYERS:-4}"
  export MOLPILOT_ALIGN_BATCH_SIZE="${MOLPILOT_ALIGN_BATCH_SIZE:-1024}"
  export MOLPILOT_DIFFUSION_HIDDEN_DIM="${MOLPILOT_DIFFUSION_HIDDEN_DIM:-1024}"
  export MOLPILOT_DIFFUSION_LAYERS="${MOLPILOT_DIFFUSION_LAYERS:-6}"
  export MOLPILOT_DIFFUSION_BATCH_SIZE="${MOLPILOT_DIFFUSION_BATCH_SIZE:-1024}"
  export MOLPILOT_TIMESTEPS="${MOLPILOT_TIMESTEPS:-100}"
fi

SUBMITTED=0
for SLURM_GPUS in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$SLURM_GPUS"
  if sbatch \
    --gpus="$SLURM_GPUS" \
    --mem-per-cpu="$SLURM_MEM_PER_CPU" \
    scripts/run_staged_server.slurm.sh; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: none of the GPU candidates were accepted by Slurm."
  echo "Try checking available names with:"
  echo "  sinfo -o '%G %N' | sort -u"
  echo "Then rerun with:"
  echo "  MOLPILOT_SLURM_GPUS=<available_gpu_name>:1 bash scripts/submit_chembl_staged.sh"
  exit 2
fi
