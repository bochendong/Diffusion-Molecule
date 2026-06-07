#!/usr/bin/env bash
# Submit unified alignment + edit-token + latent-diffusion training to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATASET_PROJECT_DIR="$REPO_DIR/SketchMol-MultiProperty-EditDataset"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

DATASET_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
EDIT_MANIFEST="${SUCC_EDIT_MANIFEST:-$DATASET_OUTPUT_DIR/diffusion_edit_manifest.csv}"
UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_generation_3m_edit_v1}"
SUBMIT_DATASET_BUILD="${SMMED_SUBMIT_DATASET_BUILD:-1}"

export SMMED_RENDER_IMAGES="${SMMED_RENDER_IMAGES:-0}"
export SMMED_OUTPUT_DIR="$DATASET_OUTPUT_DIR"
export SUCC_EDIT_MANIFEST="$EDIT_MANIFEST"
export SUCC_UNIFIED_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR"
export SUCC_3M_ROOT="${SUCC_3M_ROOT:-Research/Molecule Generation/3M-Diffusion}"
export SUCC_AUTO_CLONE_3M="${SUCC_AUTO_CLONE_3M:-1}"
export SUCC_DESCRIPTION_LIMIT="${SUCC_DESCRIPTION_LIMIT:-5000}"
export SUCC_EDIT_LIMIT="${SUCC_EDIT_LIMIT:-50000}"
export SUCC_TRAIN_LIMIT="${SUCC_TRAIN_LIMIT:-50000}"
export SUCC_BATCH_SIZE="${SUCC_BATCH_SIZE:-64}"
export SUCC_EPOCHS="${SUCC_EPOCHS:-5}"
export SUCC_EVAL_LIMIT="${SUCC_EVAL_LIMIT:-1000}"
export SUCC_EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
export SUCC_EVAL_SAMPLE_STEPS="${SUCC_EVAL_SAMPLE_STEPS:-20}"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

BUILD_ACCOUNT="${SMMED_SLURM_ACCOUNT:-def-hup-ab_gpu}"
BUILD_TIME="${SMMED_SLURM_TIME:-02:00:00}"
BUILD_MEM="${SMMED_SLURM_MEM:-8G}"
BUILD_CPUS="${SMMED_SLURM_CPUS:-1}"
BUILD_JOB_NAME="${SMMED_SLURM_JOB_NAME:-smmed-build}"
BUILD_LOG_DIR="${SMMED_LOG_DIR:-$DATASET_PROJECT_DIR/logs}"

TRAIN_ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TRAIN_TIME="${SUCC_SLURM_TIME:-24:00:00}"
TRAIN_MEM="${SUCC_SLURM_MEM:-80G}"
TRAIN_CPUS="${SUCC_SLURM_CPUS:-8}"
TRAIN_JOB_NAME="${SUCC_SLURM_JOB_NAME:-succ-unified-gen}"
TRAIN_LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
TRAIN_PARTITION="${SUCC_SLURM_PARTITION:-}"
GPU_PROFILE="${SUCC_GPU_PROFILE:-h100_40gb_mig}"

mkdir -p "$BUILD_LOG_DIR" "$TRAIN_LOG_DIR"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  echo "       Set SUCC_PYTHON_BIN to a Python with torch and numpy." >&2
  exit 2
fi

if [[ -n "${SUCC_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
elif [[ "$GPU_PROFILE" == "t4" ]]; then
  GPU_CANDIDATES=("t4:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

echo "Submitting unified Understanding + latent diffusion pipeline"
echo "  dataset_output_dir=$DATASET_OUTPUT_DIR"
echo "  edit_manifest=$EDIT_MANIFEST"
echo "  unified_output_dir=$UNIFIED_OUTPUT_DIR"
echo "  3m_root=$SUCC_3M_ROOT"
echo "  auto_clone_3m=$SUCC_AUTO_CLONE_3M"
echo "  python=$SUCC_PYTHON_BIN"
echo "  description_limit=$SUCC_DESCRIPTION_LIMIT"
echo "  edit_limit=$SUCC_EDIT_LIMIT"
echo "  train_limit=$SUCC_TRAIN_LIMIT"
echo "  epochs=$SUCC_EPOCHS"
echo "  eval_limit=$SUCC_EVAL_LIMIT"
echo "  eval_sample_steps=$SUCC_EVAL_SAMPLE_STEPS"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"

build_job_id=""
if [[ "$SUBMIT_DATASET_BUILD" == "1" ]]; then
  build_output="$(
    sbatch \
      --account="$BUILD_ACCOUNT" \
      --job-name="$BUILD_JOB_NAME" \
      --time="$BUILD_TIME" \
      --mem="$BUILD_MEM" \
      --cpus-per-task="$BUILD_CPUS" \
      --output="$BUILD_LOG_DIR/%x-%j.log" \
      --export=ALL \
      --wrap="bash '$DATASET_PROJECT_DIR/scripts/run_build_dataset.sh'"
  )"
  echo "$build_output"
  build_job_id="$(echo "$build_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$build_job_id" ]]; then
    echo "ERROR: could not parse dataset build job id from sbatch output." >&2
    exit 1
  fi
else
  echo "Skipping dataset build submission because SMMED_SUBMIT_DATASET_BUILD=$SUBMIT_DATASET_BUILD"
fi

TRAIN_SBATCH_ARGS=(
  --account="$TRAIN_ACCOUNT"
  --job-name="$TRAIN_JOB_NAME"
  --time="$TRAIN_TIME"
  --mem="$TRAIN_MEM"
  --cpus-per-task="$TRAIN_CPUS"
  --output="$TRAIN_LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$TRAIN_PARTITION" ]]; then
  TRAIN_SBATCH_ARGS+=(--partition="$TRAIN_PARTITION")
fi
if [[ -n "$build_job_id" ]]; then
  TRAIN_SBATCH_ARGS+=(--dependency="afterok:$build_job_id")
fi

train_job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if train_output="$(
    sbatch "${TRAIN_SBATCH_ARGS[@]}" \
      --gpus="$GPU_REQUEST" \
      --wrap="bash '$PROJECT_DIR/scripts/run_unified_generation_smoke.sh'"
  )"; then
    echo "$train_output"
    train_job_id="$(echo "$train_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done

if [[ -z "$train_job_id" ]]; then
  echo "ERROR: failed to submit unified training job with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi

echo
echo "Pipeline submitted."
if [[ -n "$build_job_id" ]]; then
  echo "  dataset_job=$build_job_id"
fi
echo "  unified_job=$train_job_id"
echo "  dataset_summary=$UNIFIED_OUTPUT_DIR/dataset/summary.json"
echo "  alignment_checkpoint=$UNIFIED_OUTPUT_DIR/alignment/alignment_model.pt"
echo "  connector_checkpoint=$UNIFIED_OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt"
echo "  diffusion_checkpoint=$UNIFIED_OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt"
echo "  eval_metrics=$UNIFIED_OUTPUT_DIR/eval_latent/metrics.json"
