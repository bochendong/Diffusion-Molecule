#!/usr/bin/env bash
# Submit Stage 3 diffusion refine + latent eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SMU3M_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SMU3M_GPU_PROFILE="${SMU3M_GPU_PROFILE:-h100_40gb_mig}"
export SMU3M_RESOURCE_PROFILE="${SMU3M_RESOURCE_PROFILE:-throughput_40gb}"
if [[ -n "${SMU3M_DIFFUSION_EPOCHS:-}" ]]; then
  export SMU3M_DIFFUSION_EPOCHS
fi
export SMU3M_DIFFUSION_EXTRA_EPOCHS="${SMU3M_DIFFUSION_EXTRA_EPOCHS:-100}"
export SMU3M_DIFFUSION_LR="${SMU3M_DIFFUSION_LR:-3e-4}"
export SMU3M_TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
export SMU3M_EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-0}"
export SMU3M_MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_MAX_EVAL_PER_PROPERTY_COUNT:-250}"
export SMU3M_EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
export SMU3M_EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
export SMU3M_DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
export SMU3M_DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
export SMU3M_DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
export SMU3M_PRIOR_LOSS_WEIGHT="${SMU3M_PRIOR_LOSS_WEIGHT:-0.25}"
export SMU3M_TRAIN_DIFFUSION_CONNECTOR="${SMU3M_TRAIN_DIFFUSION_CONNECTOR:-1}"
export SMU3M_RUN_MATERIALIZED_BENCHMARK="${SMU3M_RUN_MATERIALIZED_BENCHMARK:-0}"
export SMU3M_RESUME="${SMU3M_RESUME:-1}"
export SMU3M_REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
export SMU3M_CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"

case "$SMU3M_RESOURCE_PROFILE" in
  economy)
    export SMU3M_BATCH_SIZE="${SMU3M_BATCH_SIZE:-512}"
    export SMU3M_EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-512}"
    export SMU3M_NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
    export SMU3M_PIN_MEMORY="${SMU3M_PIN_MEMORY:-1}"
    export SMU3M_DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
    export SMU3M_DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
    DEFAULT_TRAIN_CPUS="2"
    DEFAULT_TRAIN_MEM="16G"
    DEFAULT_TRAIN_TIME="02:00:00"
    ;;
  throughput_40gb)
    export SMU3M_BATCH_SIZE="${SMU3M_BATCH_SIZE:-4096}"
    export SMU3M_EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-4096}"
    export SMU3M_NUM_WORKERS="${SMU3M_NUM_WORKERS:-2}"
    export SMU3M_PIN_MEMORY="${SMU3M_PIN_MEMORY:-1}"
    export SMU3M_DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-1024}"
    export SMU3M_DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-6}"
    DEFAULT_TRAIN_CPUS="4"
    DEFAULT_TRAIN_MEM="40G"
    DEFAULT_TRAIN_TIME="04:00:00"
    ;;
  *)
    echo "ERROR: unsupported SMU3M_RESOURCE_PROFILE=$SMU3M_RESOURCE_PROFILE" >&2
    exit 2
    ;;
esac

TRAIN_ACCOUNT="${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TRAIN_TIME="${SMU3M_SLURM_TIME:-$DEFAULT_TRAIN_TIME}"
TRAIN_MEM="${SMU3M_SLURM_MEM:-$DEFAULT_TRAIN_MEM}"
TRAIN_CPUS="${SMU3M_SLURM_CPUS:-$DEFAULT_TRAIN_CPUS}"
TRAIN_JOB_NAME="${SMU3M_SLURM_JOB_NAME:-smu3m-diff-refine}"
TRAIN_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
TRAIN_PARTITION="${SMU3M_SLURM_PARTITION:-}"

if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SMU3M_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SLURM_GPUS")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
else
  GPU_CANDIDATES=("$SMU3M_GPU_PROFILE")
fi

mkdir -p "$TRAIN_LOG_DIR"

echo "Submitting Unified 3M diffusion refine"
echo "  output_dir=$SMU3M_OUTPUT_DIR"
echo "  resource_profile=$SMU3M_RESOURCE_PROFILE"
echo "  diffusion_epochs=${SMU3M_DIFFUSION_EPOCHS:-auto_current_plus_extra}"
echo "  diffusion_extra_epochs=$SMU3M_DIFFUSION_EXTRA_EPOCHS"
echo "  diffusion_lr=$SMU3M_DIFFUSION_LR"
echo "  prior_loss_weight=$SMU3M_PRIOR_LOSS_WEIGHT"
echo "  train_diffusion_connector=$SMU3M_TRAIN_DIFFUSION_CONNECTOR"
echo "  batch_size=$SMU3M_BATCH_SIZE"
echo "  eval_limit=$SMU3M_EVAL_LIMIT"
echo "  max_eval_per_property_count=$SMU3M_MAX_EVAL_PER_PROPERTY_COUNT"
echo "  run_materialized_benchmark=$SMU3M_RUN_MATERIALIZED_BENCHMARK"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  train_cpus=$TRAIN_CPUS"
echo "  train_mem=$TRAIN_MEM"
echo "  train_time=$TRAIN_TIME"

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

train_job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if train_output="$(
    sbatch "${TRAIN_SBATCH_ARGS[@]}" \
      --gpus="$GPU_REQUEST" \
      --wrap="bash '$PROJECT_DIR/scripts/run_unified_diffusion_refine.sh'"
  )"; then
    echo "$train_output"
    train_job_id="$(echo "$train_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done

if [[ -z "$train_job_id" ]]; then
  echo "ERROR: failed to submit diffusion refine job." >&2
  exit 1
fi

echo
echo "Diffusion refine submitted."
echo "  job_id=$train_job_id"
echo "  log=$TRAIN_LOG_DIR/${TRAIN_JOB_NAME}-${train_job_id}.log"
echo "  diffusion_checkpoint=$SMU3M_OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt"
echo "  eval_metrics=$SMU3M_OUTPUT_DIR/eval_latent/metrics.json"
if [[ "$SMU3M_RUN_MATERIALIZED_BENCHMARK" == "1" ]]; then
  echo "  benchmark_report=$SMU3M_OUTPUT_DIR/benchmark_materialized/benchmark_report.md"
else
  echo "  benchmark_next=bash $PROJECT_DIR/scripts/submit_unified_materialized_benchmark.sh"
fi
