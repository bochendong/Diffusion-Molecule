#!/bin/bash
# Submit Phase 4A source-conditioned edit/action planner.

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
  SKETCHIMAGE_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python bash scripts/submit_phase4_edit_action_planner.sh
EOF
  exit 2
fi

export SKETCHIMAGE_SEED="${SKETCHIMAGE_SEED:-7}"
export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase4_edit_action_planner_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_ORACLE_DECODER_DIR="${SKETCHIMAGE_ORACLE_DECODER_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_TRAIN_CSV="${SKETCHIMAGE_TRAIN_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_train.csv}"
export SKETCHIMAGE_EVAL_CSV="${SKETCHIMAGE_EVAL_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_eval.csv}"
export SKETCHIMAGE_TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-35}"
export SKETCHIMAGE_TORCH_BATCH_SIZE="${SKETCHIMAGE_TORCH_BATCH_SIZE:-128}"
export SKETCHIMAGE_TORCH_HIDDEN_DIM="${SKETCHIMAGE_TORCH_HIDDEN_DIM:-1024}"
export SKETCHIMAGE_SAMPLES_PER_ALPHA="${SKETCHIMAGE_SAMPLES_PER_ALPHA:-2}"
export SKETCHIMAGE_ACTION_ALPHAS="${SKETCHIMAGE_ACTION_ALPHAS:-0.25,0.50,0.75,1.00,1.25}"
export SKETCHIMAGE_ACTION_TARGET_MODE="${SKETCHIMAGE_ACTION_TARGET_MODE:-raw_delta}"
export SKETCHIMAGE_ACTION_STEP_MODE="${SKETCHIMAGE_ACTION_STEP_MODE:-implicit}"
export SKETCHIMAGE_ACTION_STEP_RIDGE="${SKETCHIMAGE_ACTION_STEP_RIDGE:-0.01}"
export SKETCHIMAGE_ACTION_STEP_CLIP_QUANTILE="${SKETCHIMAGE_ACTION_STEP_CLIP_QUANTILE:-0.98}"
export SKETCHIMAGE_RENDER_IMAGE_CONTEXT="${SKETCHIMAGE_RENDER_IMAGE_CONTEXT:-1}"

if [[ ! -e "$SKETCHIMAGE_ORACLE_DECODER_DIR/model.pt" && ! -e "$SKETCHIMAGE_ORACLE_DECODER_DIR/model/model.pt" ]]; then
  echo "ERROR: Phase 1 decoder model not found under $SKETCHIMAGE_ORACLE_DECODER_DIR" >&2
  echo "Expected either $SKETCHIMAGE_ORACLE_DECODER_DIR/model.pt or $SKETCHIMAGE_ORACLE_DECODER_DIR/model/model.pt" >&2
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

SLURM_TIME="${SKETCHIMAGE_SLURM_TIME:-10:00:00}"
SLURM_MEM="${SKETCHIMAGE_SLURM_MEM:-64G}"
SLURM_CPUS="${SKETCHIMAGE_SLURM_CPUS:-8}"

echo "Submitting ${SKETCHIMAGE_PHASE_TITLE:-Phase 4A edit/action planner}:"
echo "  run_name=$SKETCHIMAGE_RUN_NAME"
echo "  oracle_decoder_dir=$SKETCHIMAGE_ORACLE_DECODER_DIR"
echo "  train_csv=$SKETCHIMAGE_TRAIN_CSV"
echo "  eval_csv=$SKETCHIMAGE_EVAL_CSV"
echo "  python=$SKETCHIMAGE_PYTHON_BIN"
echo "  epochs=$SKETCHIMAGE_TORCH_EPOCHS"
echo "  batch_size=$SKETCHIMAGE_TORCH_BATCH_SIZE"
echo "  hidden_dim=$SKETCHIMAGE_TORCH_HIDDEN_DIM"
echo "  samples_per_alpha=$SKETCHIMAGE_SAMPLES_PER_ALPHA"
echo "  action_alphas=$SKETCHIMAGE_ACTION_ALPHAS"
echo "  action_target_mode=$SKETCHIMAGE_ACTION_TARGET_MODE"
echo "  action_step_mode=$SKETCHIMAGE_ACTION_STEP_MODE"
echo "  action_step_ridge=$SKETCHIMAGE_ACTION_STEP_RIDGE"
echo "  action_step_clip_quantile=$SKETCHIMAGE_ACTION_STEP_CLIP_QUANTILE"
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
    --output="./outputs/logs/sketchimage-phase4-%j.log" \
    --error="./outputs/logs/sketchimage-phase4-%j.log" \
    scripts/run_phase4_edit_action_planner.slurm.sh; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: none of the GPU candidates were accepted by Slurm." >&2
  echo "Try SKETCHIMAGE_SLURM_GPUS=<available_gpu_name>:1 bash scripts/submit_phase4_edit_action_planner.sh" >&2
  exit 2
fi
