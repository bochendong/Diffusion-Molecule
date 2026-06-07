#!/usr/bin/env bash
# Submit packed source-aware follow-up experiments as one Slurm allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SMU3M_FOLLOWUP_OUTPUT_ROOT="${SMU3M_FOLLOWUP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1}"
export SMU3M_SWEEP_OUTPUT_ROOT="${SMU3M_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2}"
export SMU3M_FOLLOWUP_PLAN="${SMU3M_FOLLOWUP_PLAN:-all}"
export SMU3M_FOLLOWUP_LAUNCHER="${SMU3M_FOLLOWUP_LAUNCHER:-glost}"
export SMU3M_FOLLOWUP_CONCURRENCY="${SMU3M_FOLLOWUP_CONCURRENCY:-1}"
export SMU3M_FOLLOWUP_RESUME="${SMU3M_FOLLOWUP_RESUME:-0}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

# Match the short-yet-informative source-aware sweep profile by default.
export SMU3M_RESOURCE_PROFILE="${SMU3M_RESOURCE_PROFILE:-economy}"
export SMU3M_GPU_PROFILE="${SMU3M_GPU_PROFILE:-h100_20gb_mig}"
export SMU3M_DESCRIPTION_LIMIT="${SMU3M_DESCRIPTION_LIMIT:-5000}"
export SMU3M_EDIT_LIMIT="${SMU3M_EDIT_LIMIT:-50000}"
export SMU3M_TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
export SMU3M_EPOCHS="${SMU3M_EPOCHS:-5}"
export SMU3M_EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
export SMU3M_EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
export SMU3M_DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
export SMU3M_DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
export SMU3M_DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
export SMU3M_DIFFUSION_LR="${SMU3M_DIFFUSION_LR:-1e-3}"
export SMU3M_REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
export SMU3M_CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"
export SMMED_SUBMIT_DATASET_BUILD="${SMMED_SUBMIT_DATASET_BUILD:-0}"

case "$SMU3M_RESOURCE_PROFILE" in
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
    DEFAULT_CPUS="2"
    DEFAULT_MEM_PER_CPU="8G"
    DEFAULT_TIME="04:00:00"
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
    DEFAULT_CPUS="4"
    DEFAULT_MEM_PER_CPU="10G"
    DEFAULT_TIME="06:00:00"
    ;;
  *)
    echo "ERROR: unsupported SMU3M_RESOURCE_PROFILE=$SMU3M_RESOURCE_PROFILE" >&2
    exit 2
    ;;
esac

FOLLOWUP_ACCOUNT="${SMU3M_FOLLOWUP_SLURM_ACCOUNT:-${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
FOLLOWUP_TIME="${SMU3M_FOLLOWUP_SLURM_TIME:-$DEFAULT_TIME}"
FOLLOWUP_CPUS="${SMU3M_FOLLOWUP_SLURM_CPUS:-$DEFAULT_CPUS}"
FOLLOWUP_MEM_PER_CPU="${SMU3M_FOLLOWUP_SLURM_MEM_PER_CPU:-$DEFAULT_MEM_PER_CPU}"
FOLLOWUP_JOB_NAME="${SMU3M_FOLLOWUP_SLURM_JOB_NAME:-smu3m-src-follow}"
FOLLOWUP_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
FOLLOWUP_PARTITION="${SMU3M_FOLLOWUP_SLURM_PARTITION:-${SMU3M_SLURM_PARTITION:-}}"

if (( SMU3M_FOLLOWUP_CONCURRENCY <= 0 )); then
  echo "ERROR: SMU3M_FOLLOWUP_CONCURRENCY must be positive, got $SMU3M_FOLLOWUP_CONCURRENCY" >&2
  exit 2
fi
if (( FOLLOWUP_CPUS < SMU3M_FOLLOWUP_CONCURRENCY )); then
  echo "ERROR: SMU3M_FOLLOWUP_SLURM_CPUS=$FOLLOWUP_CPUS is less than SMU3M_FOLLOWUP_CONCURRENCY=$SMU3M_FOLLOWUP_CONCURRENCY" >&2
  exit 2
fi
if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SMU3M_FOLLOWUP_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_FOLLOWUP_SLURM_GPUS")
elif [[ -n "${SMU3M_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SLURM_GPUS")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
else
  GPU_CANDIDATES=("$SMU3M_GPU_PROFILE")
fi

mkdir -p "$FOLLOWUP_LOG_DIR"

echo "Submitting Unified 3M source-aware follow-up"
echo "  output_root=$SMU3M_FOLLOWUP_OUTPUT_ROOT"
echo "  sweep_output_root=$SMU3M_SWEEP_OUTPUT_ROOT"
echo "  plan=$SMU3M_FOLLOWUP_PLAN"
echo "  launcher=$SMU3M_FOLLOWUP_LAUNCHER"
echo "  concurrency=$SMU3M_FOLLOWUP_CONCURRENCY"
echo "  resource_profile=$SMU3M_RESOURCE_PROFILE"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  time=$FOLLOWUP_TIME"
echo "  cpus=$FOLLOWUP_CPUS"
echo "  mem_per_cpu=$FOLLOWUP_MEM_PER_CPU"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"

SBATCH_ARGS=(
  --account="$FOLLOWUP_ACCOUNT"
  --job-name="$FOLLOWUP_JOB_NAME"
  --time="$FOLLOWUP_TIME"
  --nodes=1
  --ntasks-per-node="$FOLLOWUP_CPUS"
  --mem-per-cpu="$FOLLOWUP_MEM_PER_CPU"
  --output="$FOLLOWUP_LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$FOLLOWUP_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$FOLLOWUP_PARTITION")
fi

followup_job_id=""
for gpu_request in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$gpu_request"
  if output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$gpu_request" --wrap="bash '$PROJECT_DIR/scripts/run_unified_sourceaware_followup.sh'")"; then
    echo "$output"
    followup_job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done

if [[ -z "$followup_job_id" ]]; then
  echo "ERROR: failed to submit source-aware follow-up." >&2
  exit 1
fi

echo
echo "Source-aware follow-up submitted."
echo "  job_id=$followup_job_id"
echo "  log=$FOLLOWUP_LOG_DIR/${FOLLOWUP_JOB_NAME}-${followup_job_id}.log"
echo "  task_file=$SMU3M_FOLLOWUP_OUTPUT_ROOT/tasks/sourceaware_followup.tasks"
echo "  summary=$SMU3M_FOLLOWUP_OUTPUT_ROOT/followup_summary.md"
