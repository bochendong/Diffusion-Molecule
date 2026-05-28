#!/bin/bash
# Submit Phase 2C latent calibration before decoder sampling.

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
  SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python bash scripts/submit_phase2_calibrated_decoder.sh
EOF
  exit 2
fi

export SKETCHIMAGE_SEED="${SKETCHIMAGE_SEED:-7}"
export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase2_calibrated_decoder_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_DECODER_DIR="${SKETCHIMAGE_DECODER_DIR:-outputs/runs/phase2_robust_decoder_2048_seed${SKETCHIMAGE_SEED}/decoder}"
export SKETCHIMAGE_DECODER_POOL_DIR="${SKETCHIMAGE_DECODER_POOL_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_TRAIN_CSV="${SKETCHIMAGE_TRAIN_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_train.csv}"
export SKETCHIMAGE_EVAL_CSV="${SKETCHIMAGE_EVAL_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_eval.csv}"
export SKETCHIMAGE_TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-25}"
export SKETCHIMAGE_TORCH_BATCH_SIZE="${SKETCHIMAGE_TORCH_BATCH_SIZE:-128}"
export SKETCHIMAGE_TORCH_HIDDEN_DIM="${SKETCHIMAGE_TORCH_HIDDEN_DIM:-1024}"
export SKETCHIMAGE_CALIBRATION_MODE="${SKETCHIMAGE_CALIBRATION_MODE:-residual_ridge}"
export SKETCHIMAGE_CALIBRATION_RIDGE="${SKETCHIMAGE_CALIBRATION_RIDGE:-0.01}"
export SKETCHIMAGE_CALIBRATION_BLEND="${SKETCHIMAGE_CALIBRATION_BLEND:-1.0}"
export SKETCHIMAGE_CALIBRATION_NORMALIZE="${SKETCHIMAGE_CALIBRATION_NORMALIZE:-1}"
export SKETCHIMAGE_RENDER_IMAGE_CONTEXT="${SKETCHIMAGE_RENDER_IMAGE_CONTEXT:-1}"

if [[ ! -e "$SKETCHIMAGE_DECODER_DIR/model.pt" && ! -e "$SKETCHIMAGE_DECODER_DIR/model/model.pt" ]]; then
  echo "ERROR: decoder model not found under $SKETCHIMAGE_DECODER_DIR" >&2
  echo "Expected either $SKETCHIMAGE_DECODER_DIR/model.pt or $SKETCHIMAGE_DECODER_DIR/model/model.pt" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_TRAIN_CSV" || ! -f "$SKETCHIMAGE_EVAL_CSV" ]]; then
  cat <<EOF >&2
ERROR: hard split CSVs not found.
  train_csv=$SKETCHIMAGE_TRAIN_CSV
  eval_csv=$SKETCHIMAGE_EVAL_CSV

Build the hard split first, or pass SKETCHIMAGE_TRAIN_CSV and SKETCHIMAGE_EVAL_CSV explicitly.
EOF
  exit 2
fi

mkdir -p outputs/logs

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

echo "Submitting Phase 2C calibrated decoder:"
echo "  run_name=$SKETCHIMAGE_RUN_NAME"
echo "  decoder_dir=$SKETCHIMAGE_DECODER_DIR"
echo "  decoder_pool_dir=$SKETCHIMAGE_DECODER_POOL_DIR"
echo "  train_csv=$SKETCHIMAGE_TRAIN_CSV"
echo "  eval_csv=$SKETCHIMAGE_EVAL_CSV"
echo "  python=$SKETCHIMAGE_PYTHON_BIN"
echo "  planner_epochs=$SKETCHIMAGE_TORCH_EPOCHS"
echo "  planner_batch_size=$SKETCHIMAGE_TORCH_BATCH_SIZE"
echo "  planner_hidden_dim=$SKETCHIMAGE_TORCH_HIDDEN_DIM"
echo "  calibration_mode=$SKETCHIMAGE_CALIBRATION_MODE"
echo "  calibration_ridge=$SKETCHIMAGE_CALIBRATION_RIDGE"
echo "  calibration_blend=$SKETCHIMAGE_CALIBRATION_BLEND"
echo "  calibration_normalize=$SKETCHIMAGE_CALIBRATION_NORMALIZE"
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
    --output="./outputs/logs/sketchimage-phase2c-%j.log" \
    --error="./outputs/logs/sketchimage-phase2c-%j.log" \
    scripts/run_phase2_calibrated_decoder.slurm.sh; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: none of the GPU candidates were accepted by Slurm." >&2
  echo "Try SKETCHIMAGE_SLURM_GPUS=<available_gpu_name>:1 bash scripts/submit_phase2_calibrated_decoder.sh" >&2
  exit 2
fi
