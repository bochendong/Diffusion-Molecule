#!/usr/bin/env bash
# Submit Route B joint image+SMILES diffusion to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ACCOUNT="${SKETCHMOL_JOINT_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHMOL_JOINT_SLURM_TIME:-10:00:00}"
MEM="${SKETCHMOL_JOINT_SLURM_MEM:-40G}"
CPUS="${SKETCHMOL_JOINT_SLURM_CPUS:-4}"
GPU_PROFILE="${SKETCHMOL_JOINT_GPU_PROFILE:-h100_10gb_mig}"
JOB_NAME="${SKETCHMOL_JOINT_SLURM_JOB_NAME:-joint-diffusion}"
LOG_DIR="${SKETCHMOL_JOINT_LOG_DIR:-SketchMolJointDiffusion/logs}"
mkdir -p "$LOG_DIR"

export SKETCHMOL_JOINT_MODULES="${SKETCHMOL_JOINT_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHMOL_JOINT_PYTHON_BIN="${SKETCHMOL_JOINT_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
export SKETCHMOL_JOINT_PAIR_DIR="${SKETCHMOL_JOINT_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
export SKETCHMOL_JOINT_RUN_NAME="${SKETCHMOL_JOINT_RUN_NAME:-joint_diffusion_seed${SKETCHMOL_JOINT_SEED:-7}}"
export SKETCHMOL_JOINT_EPOCHS="${SKETCHMOL_JOINT_EPOCHS:-20}"
export SKETCHMOL_JOINT_BATCH_SIZE="${SKETCHMOL_JOINT_BATCH_SIZE:-128}"
export SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT="${SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT:-1.0}"
export SKETCHMOL_JOINT_DEVICE="${SKETCHMOL_JOINT_DEVICE:-auto}"

if [[ -n "${SKETCHMOL_JOINT_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SKETCHMOL_JOINT_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1" "h100_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1" "nvidia_h100_80gb_hbm3_3g.40gb:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

echo "Submitting SketchMolJointDiffusion Route B:"
echo "  run_name=$SKETCHMOL_JOINT_RUN_NAME"
echo "  pair_dir=$SKETCHMOL_JOINT_PAIR_DIR"
echo "  epochs=$SKETCHMOL_JOINT_EPOCHS"
echo "  image_loss_weight=$SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"

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
    --wrap="bash 'SketchMolJointDiffusion/scripts/run_joint_diffusion.sh'"; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi
