#!/usr/bin/env bash
# Submit Route A masked token diffusion to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ACCOUNT="${SKETCHMOL_TOKEN_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHMOL_TOKEN_SLURM_TIME:-08:00:00}"
MEM="${SKETCHMOL_TOKEN_SLURM_MEM:-32G}"
CPUS="${SKETCHMOL_TOKEN_SLURM_CPUS:-4}"
GPU_PROFILE="${SKETCHMOL_TOKEN_GPU_PROFILE:-h100_10gb_mig}"
JOB_NAME="${SKETCHMOL_TOKEN_SLURM_JOB_NAME:-token-diffusion}"
LOG_DIR="${SKETCHMOL_TOKEN_LOG_DIR:-SketchMolTokenDiffusion/logs}"
mkdir -p "$LOG_DIR"

export SKETCHMOL_TOKEN_MODULES="${SKETCHMOL_TOKEN_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHMOL_TOKEN_PYTHON_BIN="${SKETCHMOL_TOKEN_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
export SKETCHMOL_TOKEN_PAIR_DIR="${SKETCHMOL_TOKEN_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
export SKETCHMOL_TOKEN_RUN_NAME="${SKETCHMOL_TOKEN_RUN_NAME:-token_diffusion_seed${SKETCHMOL_TOKEN_SEED:-7}}"
export SKETCHMOL_TOKEN_EPOCHS="${SKETCHMOL_TOKEN_EPOCHS:-20}"
export SKETCHMOL_TOKEN_BATCH_SIZE="${SKETCHMOL_TOKEN_BATCH_SIZE:-128}"
export SKETCHMOL_TOKEN_TOKENIZATION="${SKETCHMOL_TOKEN_TOKENIZATION:-smiles_token}"
export SKETCHMOL_TOKEN_DECODE_LENGTH_MODE="${SKETCHMOL_TOKEN_DECODE_LENGTH_MODE:-free}"
export SKETCHMOL_TOKEN_DEVICE="${SKETCHMOL_TOKEN_DEVICE:-auto}"

if [[ -n "${SKETCHMOL_TOKEN_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SKETCHMOL_TOKEN_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1" "h100_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1" "nvidia_h100_80gb_hbm3_3g.40gb:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

echo "Submitting SketchMolTokenDiffusion Route A:"
echo "  run_name=$SKETCHMOL_TOKEN_RUN_NAME"
echo "  pair_dir=$SKETCHMOL_TOKEN_PAIR_DIR"
echo "  epochs=$SKETCHMOL_TOKEN_EPOCHS"
echo "  batch_size=$SKETCHMOL_TOKEN_BATCH_SIZE"
echo "  tokenization=$SKETCHMOL_TOKEN_TOKENIZATION"
echo "  decode_length_mode=$SKETCHMOL_TOKEN_DECODE_LENGTH_MODE"
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
    --wrap="bash 'SketchMolTokenDiffusion/scripts/run_masked_token_diffusion.sh'"; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi
