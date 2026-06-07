#!/usr/bin/env bash
# Submit OCR-only rerun for existing UniVideo v2_residual_ink outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TIME="${SUCC_SLURM_TIME:-02:00:00}"
MEM="${SUCC_SLURM_MEM:-32G}"
CPUS="${SUCC_SLURM_CPUS:-4}"
JOB_NAME="${SUCC_SLURM_JOB_NAME:-succ-ocr-benchmark}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_SLURM_PARTITION:-}"
GPU_PROFILE="${SUCC_GPU_PROFILE:-h100_20gb_mig}"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink}"
export SUCC_MOLSCRIBE_MODEL="${SUCC_MOLSCRIBE_MODEL:-/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth}"
export SUCC_MOLSCRIBE_WORKDIR="${SUCC_MOLSCRIBE_WORKDIR:-Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate}"
export SUCC_ONMT_OVERLAY="${SUCC_ONMT_OVERLAY:-/scratch/bdong/python_overlays/onmt220}"
export SKETCHMOL_ONMT_OVERLAY="${SKETCHMOL_ONMT_OVERLAY:-$SUCC_ONMT_OVERLAY}"
export SUCC_MOLSCRIBE_BACKEND="${SUCC_MOLSCRIBE_BACKEND:-sketchmol}"
export SUCC_MOLSCRIBE_DEVICE="${SUCC_MOLSCRIBE_DEVICE:-cuda}"
export SUCC_PREPROCESS_IMAGES="${SUCC_PREPROCESS_IMAGES:-0}"
export SUCC_RUN_MOLSCRIBE_DIAGNOSTIC="${SUCC_RUN_MOLSCRIBE_DIAGNOSTIC:-1}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting SUCC OCR benchmark rerun"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  python=$SUCC_PYTHON_BIN"
echo "  run_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  molscribe_workdir=${SUCC_MOLSCRIBE_WORKDIR}"
echo "  onmt_overlay=${SUCC_ONMT_OVERLAY}"
echo "  preprocess_images=${SUCC_PREPROCESS_IMAGES}"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if output="$(
    sbatch "${SBATCH_ARGS[@]}" \
      --gpus="$GPU_REQUEST" \
      --wrap="bash '$PROJECT_DIR/scripts/run_succ_ocr_benchmark.sh'"
  )"; then
    echo "$output"
    job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit OCR benchmark job." >&2
  exit 1
fi

echo
echo "OCR benchmark submitted."
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
echo "  report=$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/image_structure_benchmark/benchmark_report.md"
