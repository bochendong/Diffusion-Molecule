#!/usr/bin/env bash
# Submit Latent Edit Trajectory Attention synthetic training to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

ACCOUNT="${LETA_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${LETA_SLURM_TIME:-02:00:00}"
MEM="${LETA_SLURM_MEM:-24G}"
CPUS="${LETA_SLURM_CPUS:-4}"
GPU_PROFILE="${LETA_GPU_PROFILE:-h100_10gb_mig}"
JOB_NAME="${LETA_SLURM_JOB_NAME:-leta-traj-attn}"
LOG_DIR="${LETA_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export LETA_PYTHON_BIN="${LETA_PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}"
export LETA_RUN_NAME="${LETA_RUN_NAME:-trajectory_attention_seed${LETA_SEED:-7}}"
export LETA_EPOCHS="${LETA_EPOCHS:-20}"
export LETA_BATCH_SIZE="${LETA_BATCH_SIZE:-128}"
export LETA_EXAMPLES="${LETA_EXAMPLES:-4096}"
export LETA_HISTORY_LENGTH="${LETA_HISTORY_LENGTH:-8}"
export LETA_DEVICE="${LETA_DEVICE:-auto}"

if [[ -n "${LETA_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$LETA_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1" "h100_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1" "nvidia_h100_80gb_hbm3_3g.40gb:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

echo "Submitting Latent Edit Trajectory Attention:"
echo "  run_name=$LETA_RUN_NAME"
echo "  python=$LETA_PYTHON_BIN"
echo "  examples=$LETA_EXAMPLES"
echo "  history_length=$LETA_HISTORY_LENGTH"
echo "  epochs=$LETA_EPOCHS"
echo "  batch_size=$LETA_BATCH_SIZE"
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
    --wrap="bash '$PROJECT_DIR/scripts/run_synthetic_training.sh'"; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi

