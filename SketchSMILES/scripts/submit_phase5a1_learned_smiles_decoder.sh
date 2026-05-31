#!/usr/bin/env bash
# Submit Phase 5A-1 oracle-conditioned learned SMILES decoder to Slurm.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACCOUNT="${SKETCHSMILES_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHSMILES_SLURM_TIME:-04:00:00}"
MEM="${SKETCHSMILES_SLURM_MEM:-32G}"
CPUS="${SKETCHSMILES_SLURM_CPUS:-4}"
GPU_PROFILE="${SKETCHSMILES_GPU_PROFILE:-h100_10gb_mig}"
JOB_NAME="${SKETCHSMILES_SLURM_JOB_NAME:-sketchsmiles-5a1}"
LOG_DIR="${SKETCHSMILES_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export SKETCHSMILES_MODULES="${SKETCHSMILES_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHSMILES_PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
export SKETCHSMILES_PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
export SKETCHSMILES_RUN_NAME="${SKETCHSMILES_RUN_NAME:-phase5a1_learned_smiles_decoder_seed${SKETCHSMILES_SEED:-7}}"
export SKETCHSMILES_EPOCHS="${SKETCHSMILES_EPOCHS:-20}"
export SKETCHSMILES_BATCH_SIZE="${SKETCHSMILES_BATCH_SIZE:-128}"
export SKETCHSMILES_DEVICE="${SKETCHSMILES_DEVICE:-auto}"

if [[ -n "${SKETCHSMILES_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SKETCHSMILES_SLURM_GPUS")
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

echo "Submitting SketchSMILES Phase 5A-1:"
echo "  run_name=$SKETCHSMILES_RUN_NAME"
echo "  pair_dir=$SKETCHSMILES_PAIR_DIR"
echo "  python=$SKETCHSMILES_PYTHON_BIN"
echo "  epochs=$SKETCHSMILES_EPOCHS"
echo "  batch_size=$SKETCHSMILES_BATCH_SIZE"
echo "  gpu_profile=$GPU_PROFILE"
echo "  slurm_gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  slurm_time=$TIME"
echo "  slurm_mem=$MEM"
echo "  slurm_cpus=$CPUS"

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
    --wrap="bash scripts/run_phase5a1_learned_smiles_decoder.sh"; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi
