#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA="${MOLPILOT_DATA:-../PhysTabMol/data/molecules.csv}"
LIMIT="${MOLPILOT_LIMIT:-10000}"
EVAL_LIMIT="${MOLPILOT_EVAL_LIMIT:-1000}"
EVAL_MOLECULE_LIMIT="${MOLPILOT_EVAL_MOLECULE_LIMIT:-$LIMIT}"
CODEC="${MOLPILOT_CODEC:-sequence}"
RUN_NAME="${MOLPILOT_RUN_NAME:-molpilot_${CODEC}_${LIMIT}_$(date +%Y%m%d_%H%M%S)}"
TASK_MODE="${MOLPILOT_TASK_MODE:-verified}"
GPU_PROFILE="${MOLPILOT_GPU_PROFILE:-h100_40gb_mig}"
SLURM_MEM_PER_CPU="${MOLPILOT_SLURM_MEM_PER_CPU:-4096M}"
SLURM_TIME="${MOLPILOT_SLURM_TIME:-02:00:00}"
STAGE2_MODEL="${MOLPILOT_STAGE2_MODEL:-jepa}"
DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    PYTHON_BIN="$(command -v python || command -v python3)"
  fi
fi

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

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: could not find an executable Python. Activate your venv or set PYTHON_BIN."
  exit 2
fi

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN cannot import numpy."
  echo "Activate the phystabmol venv first, or run:"
  echo "  PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python bash scripts/submit_chembl_staged.sh"
  exit 2
fi

echo "Submitting MolPilot staged ChEMBL run:"
echo "  data=$DATA"
echo "  limit=$LIMIT"
echo "  eval_limit=$EVAL_LIMIT"
echo "  eval_molecule_limit=$EVAL_MOLECULE_LIMIT"
echo "  codec=$CODEC"
echo "  gpu_profile=$GPU_PROFILE"
echo "  stage2_model=$STAGE2_MODEL"
echo "  task_mode=$TASK_MODE"
echo "  slurm_gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  slurm_mem_per_cpu=$SLURM_MEM_PER_CPU"
echo "  slurm_time=$SLURM_TIME"
echo "  python_bin=$PYTHON_BIN"
echo "  run_name=$RUN_NAME"

export MOLPILOT_DATA="$DATA"
export MOLPILOT_LIMIT="$LIMIT"
export MOLPILOT_EVAL_LIMIT="$EVAL_LIMIT"
export MOLPILOT_EVAL_MOLECULE_LIMIT="$EVAL_MOLECULE_LIMIT"
export MOLPILOT_CODEC="$CODEC"
export MOLPILOT_RUN_NAME="$RUN_NAME"
export MOLPILOT_STAGE2_MODEL="$STAGE2_MODEL"
export MOLPILOT_TASK_MODE="$TASK_MODE"
export PYTHON_BIN="$PYTHON_BIN"
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
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  export MOLPILOT_EMBEDDING_DIM="${MOLPILOT_EMBEDDING_DIM:-160}"
  export MOLPILOT_MAX_LENGTH="${MOLPILOT_MAX_LENGTH:-112}"
  export MOLPILOT_AE_HIDDEN_DIM="${MOLPILOT_AE_HIDDEN_DIM:-512}"
  export MOLPILOT_AE_LAYERS="${MOLPILOT_AE_LAYERS:-3}"
  export MOLPILOT_AE_BATCH_SIZE="${MOLPILOT_AE_BATCH_SIZE:-256}"
  export MOLPILOT_ALIGN_HIDDEN_DIM="${MOLPILOT_ALIGN_HIDDEN_DIM:-640}"
  export MOLPILOT_ALIGN_LAYERS="${MOLPILOT_ALIGN_LAYERS:-3}"
  export MOLPILOT_ALIGN_BATCH_SIZE="${MOLPILOT_ALIGN_BATCH_SIZE:-768}"
  export MOLPILOT_DIFFUSION_HIDDEN_DIM="${MOLPILOT_DIFFUSION_HIDDEN_DIM:-768}"
  export MOLPILOT_DIFFUSION_LAYERS="${MOLPILOT_DIFFUSION_LAYERS:-5}"
  export MOLPILOT_DIFFUSION_BATCH_SIZE="${MOLPILOT_DIFFUSION_BATCH_SIZE:-768}"
  export MOLPILOT_TIMESTEPS="${MOLPILOT_TIMESTEPS:-75}"
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
    --export=ALL \
    --gpus="$SLURM_GPUS" \
    --mem-per-cpu="$SLURM_MEM_PER_CPU" \
    --time="$SLURM_TIME" \
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
