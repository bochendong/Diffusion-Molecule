#!/usr/bin/env bash
# Submit SketchMol opt-pair training to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

ACCOUNT="${LETA_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${LETA_SLURM_TIME:-00:30:00}"
MEM="${LETA_SLURM_MEM:-8G}"
CPUS="${LETA_SLURM_CPUS:-2}"
USE_GPU="${LETA_USE_GPU:-0}"
GPU_PROFILE="${LETA_GPU_PROFILE:-h100_10gb_mig}"
JOB_NAME="${LETA_SLURM_JOB_NAME:-leta-sketchmol-opt}"
LOG_DIR="${LETA_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export LETA_PYTHON_BIN="${LETA_PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}"
export LETA_RUN_NAME="${LETA_RUN_NAME:-sketchmol_opt_pairs_seed${LETA_SEED:-7}}"
export LETA_MODEL_KIND="${LETA_MODEL_KIND:-history}"
export LETA_EPOCHS="${LETA_EPOCHS:-30}"
export LETA_BATCH_SIZE="${LETA_BATCH_SIZE:-64}"
export LETA_LATENT_DIM="${LETA_LATENT_DIM:-256}"
export LETA_MODULES="${LETA_MODULES:-gcc rdkit/2025.09.4}"

if [[ -n "${LETA_SLURM_GPUS:-}" || "$USE_GPU" == "1" ]]; then
  export LETA_DEVICE="${LETA_DEVICE:-auto}"
  USE_GPU=1
else
  export LETA_DEVICE="${LETA_DEVICE:-cpu}"
fi

if [[ "$USE_GPU" == "1" && -n "${LETA_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$LETA_SLURM_GPUS")
elif [[ "$USE_GPU" == "1" && "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1" "h100_1g.10gb:1")
elif [[ "$USE_GPU" == "1" && "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$USE_GPU" == "1" && "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1" "nvidia_h100_80gb_hbm3_3g.40gb:1")
elif [[ "$USE_GPU" == "1" ]]; then
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

echo "Submitting Latent Edit Trajectory Attention SketchMol opt-pair training:"
echo "  run_name=$LETA_RUN_NAME"
echo "  model_kind=$LETA_MODEL_KIND"
echo "  python=$LETA_PYTHON_BIN"
echo "  modules=$LETA_MODULES"
echo "  epochs=$LETA_EPOCHS"
echo "  batch_size=$LETA_BATCH_SIZE"
echo "  latent_dim=$LETA_LATENT_DIM"
echo "  device=$LETA_DEVICE"
echo "  use_gpu=$USE_GPU"
if [[ "$USE_GPU" == "1" ]]; then
  echo "  gpu_candidates=${GPU_CANDIDATES[*]}"
fi

if [[ "$USE_GPU" != "1" ]]; then
  echo "Trying CPU sbatch"
  sbatch \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="bash '$PROJECT_DIR/scripts/run_sketchmol_opt_training.sh'"
else
  SUBMITTED=0
  for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
    echo "Trying sbatch with --gpus=$GPU_REQUEST"
    if sbatch \
      --account="$ACCOUNT" \
      --job-name="$JOB_NAME" \
      --time="$TIME" \
      --mem="$MEM" \
      --cpus-per-task="$CPUS" \
      --gpus="$GPU_REQUEST" \
      --output="$LOG_DIR/%x-%j.log" \
      --export=ALL \
      --wrap="bash '$PROJECT_DIR/scripts/run_sketchmol_opt_training.sh'"; then
      SUBMITTED=1
      break
    fi
  done

  if [[ "$SUBMITTED" != "1" ]]; then
    echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
    exit 1
  fi
fi

