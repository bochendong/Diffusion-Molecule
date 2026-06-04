#!/usr/bin/env bash
# Submit saved-model candidate expansion to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ACCOUNT="${SKETCHMOL_E2E_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHMOL_E2E_SLURM_TIME:-04:00:00}"
MEM="${SKETCHMOL_E2E_SLURM_MEM:-32G}"
CPUS="${SKETCHMOL_E2E_SLURM_CPUS:-4}"
GPU_PROFILE="${SKETCHMOL_E2E_GPU_PROFILE:-h100_10gb_mig}"
JOB_NAME="${SKETCHMOL_E2E_SLURM_JOB_NAME:-sketchmol-eval}"
LOG_DIR="${SKETCHMOL_E2E_LOG_DIR:-SketchMolEnd2End/logs}"
mkdir -p "$LOG_DIR"

export SKETCHMOL_E2E_MODULES="${SKETCHMOL_E2E_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHMOL_E2E_PYTHON_BIN="${SKETCHMOL_E2E_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
export SKETCHMOL_E2E_RUN_DIR="${SKETCHMOL_E2E_RUN_DIR:-SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7}"
export SKETCHMOL_E2E_PAIR_DIR="${SKETCHMOL_E2E_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
export SKETCHMOL_E2E_BEAM_SIZE="${SKETCHMOL_E2E_BEAM_SIZE:-32}"
export SKETCHMOL_E2E_DECODING="${SKETCHMOL_E2E_DECODING:-beam}"
export SKETCHMOL_E2E_RERANK_MODE="${SKETCHMOL_E2E_RERANK_MODE:-beam}"
export SKETCHMOL_E2E_OUTPUT_DIR="${SKETCHMOL_E2E_OUTPUT_DIR:-${SKETCHMOL_E2E_RUN_DIR}_eval_${SKETCHMOL_E2E_DECODING}_beam${SKETCHMOL_E2E_BEAM_SIZE}}"
export SKETCHMOL_E2E_IMAGE_SIZE="${SKETCHMOL_E2E_IMAGE_SIZE:-128}"
export SKETCHMOL_E2E_DEVICE="${SKETCHMOL_E2E_DEVICE:-auto}"

if [[ -n "${SKETCHMOL_E2E_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SKETCHMOL_E2E_SLURM_GPUS")
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

echo "Submitting SketchMolEnd2End saved-model eval:"
echo "  run_dir=$SKETCHMOL_E2E_RUN_DIR"
echo "  output_dir=$SKETCHMOL_E2E_OUTPUT_DIR"
echo "  beam_size=$SKETCHMOL_E2E_BEAM_SIZE"
echo "  decoding=$SKETCHMOL_E2E_DECODING"
echo "  rerank_mode=$SKETCHMOL_E2E_RERANK_MODE"
echo "  eval_limit=${SKETCHMOL_E2E_EVAL_LIMIT:-<all>}"
echo "  gpu_profile=$GPU_PROFILE"
echo "  slurm_gpu_candidates=${GPU_CANDIDATES[*]}"

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
    --wrap="bash 'SketchMolEnd2End/scripts/run_saved_image_to_structure_eval.sh'"; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi
