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
EDIT_MANIFEST="${SMU3M_EDIT_MANIFEST:-$DATASET_OUTPUT_DIR/diffusion_edit_manifest.csv}"
UNIFIED_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v1}"
SUBMIT_DATASET_BUILD="${SMMED_SUBMIT_DATASET_BUILD:-1}"

export SMMED_RENDER_IMAGES="${SMMED_RENDER_IMAGES:-0}"
export SMMED_OUTPUT_DIR="$DATASET_OUTPUT_DIR"
export SMU3M_EDIT_MANIFEST="$EDIT_MANIFEST"
export SMU3M_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR"
export SMU3M_3M_ROOT="${SMU3M_3M_ROOT:-Research/Molecule Generation/3M-Diffusion}"
export SMU3M_AUTO_CLONE_3M="${SMU3M_AUTO_CLONE_3M:-1}"
GPU_PROFILE="${SMU3M_GPU_PROFILE:-h100_20gb_mig}"
RESOURCE_PROFILE="${SMU3M_RESOURCE_PROFILE:-auto}"
if [[ "$RESOURCE_PROFILE" == "auto" ]]; then
  DIRECT_GPU_REQUEST="${SMU3M_SLURM_GPUS:-}"
  if [[ "$GPU_PROFILE" == "h100_40gb_mig" || "$GPU_PROFILE" == "h100_full" || "$GPU_PROFILE" == "a100" || "$DIRECT_GPU_REQUEST" == *"40gb"* || "$DIRECT_GPU_REQUEST" == "h100:1" || "$DIRECT_GPU_REQUEST" == "a100:1" ]]; then
    RESOURCE_PROFILE="throughput_40gb"
  else
    RESOURCE_PROFILE="economy"
  fi
fi

case "$RESOURCE_PROFILE" in
  economy)
    export SMU3M_BATCH_SIZE="${SMU3M_BATCH_SIZE:-512}"
    export SMU3M_EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-512}"
    export SMU3M_NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
    export SMU3M_PIN_MEMORY="${SMU3M_PIN_MEMORY:-1}"
    export SMU3M_ALIGNMENT_HIDDEN_DIM="${SMU3M_ALIGNMENT_HIDDEN_DIM:-512}"
    export SMU3M_EDIT_HIDDEN_DIM="${SMU3M_EDIT_HIDDEN_DIM:-512}"
    export SMU3M_DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
    export SMU3M_NUM_QUERIES="${SMU3M_NUM_QUERIES:-16}"
    export SMU3M_DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
    DEFAULT_TRAIN_CPUS="2"
    DEFAULT_TRAIN_MEM="16G"
    DEFAULT_TRAIN_TIME="08:00:00"
    ;;
  throughput_40gb)
    export SMU3M_BATCH_SIZE="${SMU3M_BATCH_SIZE:-4096}"
    export SMU3M_EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-4096}"
    export SMU3M_NUM_WORKERS="${SMU3M_NUM_WORKERS:-2}"
    export SMU3M_PIN_MEMORY="${SMU3M_PIN_MEMORY:-1}"
    export SMU3M_ALIGNMENT_HIDDEN_DIM="${SMU3M_ALIGNMENT_HIDDEN_DIM:-1024}"
    export SMU3M_EDIT_HIDDEN_DIM="${SMU3M_EDIT_HIDDEN_DIM:-1024}"
    export SMU3M_DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-1024}"
    export SMU3M_NUM_QUERIES="${SMU3M_NUM_QUERIES:-32}"
    export SMU3M_DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-6}"
    DEFAULT_TRAIN_CPUS="4"
    DEFAULT_TRAIN_MEM="40G"
    DEFAULT_TRAIN_TIME="12:00:00"
    ;;
  *)
    echo "ERROR: unsupported SMU3M_RESOURCE_PROFILE=$RESOURCE_PROFILE" >&2
    echo "       Use economy, throughput_40gb, or auto." >&2
    exit 2
    ;;
esac

export SMU3M_RESOURCE_PROFILE="$RESOURCE_PROFILE"
export SMU3M_GPU_PROFILE="$GPU_PROFILE"
export SMU3M_DESCRIPTION_LIMIT="${SMU3M_DESCRIPTION_LIMIT:-5000}"
export SMU3M_EDIT_LIMIT="${SMU3M_EDIT_LIMIT:-50000}"
export SMU3M_TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
export SMU3M_EPOCHS="${SMU3M_EPOCHS:-5}"
export SMU3M_EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-1000}"
export SMU3M_EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
export SMU3M_EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
export SMU3M_DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
export SMU3M_DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
export SMU3M_DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
export SMU3M_PRIOR_LOSS_WEIGHT="${SMU3M_PRIOR_LOSS_WEIGHT:-0.0}"
export SMU3M_DEVICE="${SMU3M_DEVICE:-auto}"
export SMU3M_REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
export SMU3M_CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"
export SMU3M_RESUME="${SMU3M_RESUME:-1}"
export SMU3M_INCLUDE_PUBCHEM="${SMU3M_INCLUDE_PUBCHEM:-1}"
export SMU3M_INCLUDE_KV="${SMU3M_INCLUDE_KV:-0}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

BUILD_ACCOUNT="${SMMED_SLURM_ACCOUNT:-def-hup-ab_gpu}"
BUILD_TIME="${SMMED_SLURM_TIME:-02:00:00}"
BUILD_MEM="${SMMED_SLURM_MEM:-8G}"
BUILD_CPUS="${SMMED_SLURM_CPUS:-1}"
BUILD_JOB_NAME="${SMMED_SLURM_JOB_NAME:-smmed-build}"
BUILD_LOG_DIR="${SMMED_LOG_DIR:-$DATASET_PROJECT_DIR/logs}"

TRAIN_ACCOUNT="${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TRAIN_TIME="${SMU3M_SLURM_TIME:-$DEFAULT_TRAIN_TIME}"
TRAIN_MEM="${SMU3M_SLURM_MEM:-$DEFAULT_TRAIN_MEM}"
TRAIN_CPUS="${SMU3M_SLURM_CPUS:-$DEFAULT_TRAIN_CPUS}"
TRAIN_JOB_NAME="${SMU3M_SLURM_JOB_NAME:-smu3m-unified}"
TRAIN_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
TRAIN_PARTITION="${SMU3M_SLURM_PARTITION:-}"

mkdir -p "$BUILD_LOG_DIR" "$TRAIN_LOG_DIR"

if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  echo "       Set SMU3M_PYTHON_BIN to a Python with torch and numpy." >&2
  exit 2
fi

if [[ -n "${SMU3M_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SLURM_GPUS")
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
echo "  resource_profile=$SMU3M_RESOURCE_PROFILE"
echo "  3m_root=$SMU3M_3M_ROOT"
echo "  auto_clone_3m=$SMU3M_AUTO_CLONE_3M"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  description_limit=$SMU3M_DESCRIPTION_LIMIT"
echo "  edit_limit=$SMU3M_EDIT_LIMIT"
echo "  train_limit=$SMU3M_TRAIN_LIMIT"
echo "  batch_size=$SMU3M_BATCH_SIZE"
echo "  eval_batch_size=$SMU3M_EVAL_BATCH_SIZE"
echo "  num_workers=$SMU3M_NUM_WORKERS"
echo "  pin_memory=$SMU3M_PIN_MEMORY"
echo "  epochs=$SMU3M_EPOCHS"
echo "  eval_limit=$SMU3M_EVAL_LIMIT"
echo "  eval_sample_steps=$SMU3M_EVAL_SAMPLE_STEPS"
echo "  eval_sample_eta=$SMU3M_EVAL_SAMPLE_ETA"
echo "  diffusion_timesteps=$SMU3M_DIFFUSION_TIMESTEPS"
echo "  diffusion_objective=$SMU3M_DIFFUSION_OBJECTIVE"
echo "  diffusion_target=$SMU3M_DIFFUSION_TARGET"
echo "  prior_loss_weight=$SMU3M_PRIOR_LOSS_WEIGHT"
echo "  device=$SMU3M_DEVICE"
echo "  require_cuda=$SMU3M_REQUIRE_CUDA"
echo "  checkpoint_every=$SMU3M_CHECKPOINT_EVERY"
echo "  resume=$SMU3M_RESUME"
echo "  include_pubchem=$SMU3M_INCLUDE_PUBCHEM"
echo "  include_kv=$SMU3M_INCLUDE_KV"
echo "  alignment_hidden_dim=$SMU3M_ALIGNMENT_HIDDEN_DIM"
echo "  edit_hidden_dim=$SMU3M_EDIT_HIDDEN_DIM"
echo "  diffusion_hidden_dim=$SMU3M_DIFFUSION_HIDDEN_DIM"
echo "  num_queries=$SMU3M_NUM_QUERIES"
echo "  diffusion_depth=$SMU3M_DIFFUSION_DEPTH"
echo "  train_cpus=$TRAIN_CPUS"
echo "  train_mem=$TRAIN_MEM"
echo "  train_time=$TRAIN_TIME"
echo "  build_cpus=$BUILD_CPUS"
echo "  build_mem=$BUILD_MEM"
echo "  build_time=$BUILD_TIME"
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
