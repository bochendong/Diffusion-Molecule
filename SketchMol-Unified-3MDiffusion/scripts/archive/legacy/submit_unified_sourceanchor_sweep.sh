#!/usr/bin/env bash
# Submit packed source-anchored fingerprint diffusion refinements.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SMU3M_BASE_OUTPUT_DIR="${SMU3M_BASE_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_sourceguard_v1}"
export SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT="${SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceanchor_sweep_v1}"
export SMU3M_SOURCEANCHOR_SWEEP_LAUNCHER="${SMU3M_SOURCEANCHOR_SWEEP_LAUNCHER:-glost}"
export SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY="${SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY:-1}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

# Match the economy sourceguard base model shape while keeping CPU requests honest.
export SMU3M_GPU_PROFILE="${SMU3M_GPU_PROFILE:-h100_20gb_mig}"
export SMU3M_BATCH_SIZE="${SMU3M_BATCH_SIZE:-512}"
export SMU3M_EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-512}"
export SMU3M_NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
export SMU3M_PIN_MEMORY="${SMU3M_PIN_MEMORY:-1}"
export SMU3M_TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
export SMU3M_DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
export SMU3M_DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
export SMU3M_DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
export SMU3M_DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
export SMU3M_DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
export SMU3M_DIFFUSION_LR="${SMU3M_DIFFUSION_LR:-3e-4}"
export SMU3M_EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
export SMU3M_EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
export SMU3M_REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
export SMU3M_CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"

export SMU3M_SOURCEANCHOR_EVAL_LIMIT="${SMU3M_SOURCEANCHOR_EVAL_LIMIT:-0}"
export SMU3M_SOURCEANCHOR_MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_SOURCEANCHOR_MAX_EVAL_PER_PROPERTY_COUNT:-250}"
export SMU3M_SOURCEANCHOR_SOURCE_REGRET_LOSS_WEIGHT="${SMU3M_SOURCEANCHOR_SOURCE_REGRET_LOSS_WEIGHT:-0.35}"
export SMU3M_SOURCEANCHOR_SOURCE_RADIUS_LOSS_WEIGHT="${SMU3M_SOURCEANCHOR_SOURCE_RADIUS_LOSS_WEIGHT:-0.10}"
export SMU3M_SOURCEANCHOR_TRAIN_DIFFUSION_CONNECTOR="${SMU3M_SOURCEANCHOR_TRAIN_DIFFUSION_CONNECTOR:-1}"

SWEEP_ACCOUNT="${SMU3M_SOURCEANCHOR_SWEEP_SLURM_ACCOUNT:-${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
SWEEP_TIME="${SMU3M_SOURCEANCHOR_SWEEP_SLURM_TIME:-04:00:00}"
SWEEP_CPUS="${SMU3M_SOURCEANCHOR_SWEEP_SLURM_CPUS:-$SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY}"
SWEEP_MEM_PER_CPU="${SMU3M_SOURCEANCHOR_SWEEP_SLURM_MEM_PER_CPU:-8G}"
SWEEP_JOB_NAME="${SMU3M_SOURCEANCHOR_SWEEP_SLURM_JOB_NAME:-smu3m-src-anchor}"
SWEEP_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
SWEEP_PARTITION="${SMU3M_SOURCEANCHOR_SWEEP_SLURM_PARTITION:-${SMU3M_SLURM_PARTITION:-}}"

if (( SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY <= 0 )); then
  echo "ERROR: SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY must be positive, got $SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY" >&2
  exit 2
fi
if (( SWEEP_CPUS < SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY )); then
  echo "ERROR: SMU3M_SOURCEANCHOR_SWEEP_SLURM_CPUS=$SWEEP_CPUS is less than SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY=$SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY" >&2
  exit 2
fi
if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SMU3M_SOURCEANCHOR_SWEEP_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SOURCEANCHOR_SWEEP_SLURM_GPUS")
elif [[ -n "${SMU3M_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SLURM_GPUS")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
else
  GPU_CANDIDATES=("$SMU3M_GPU_PROFILE")
fi

mkdir -p "$SWEEP_LOG_DIR"

echo "Submitting Unified 3M source-anchor sweep"
echo "  base_output_dir=$SMU3M_BASE_OUTPUT_DIR"
echo "  output_root=$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT"
echo "  launcher=$SMU3M_SOURCEANCHOR_SWEEP_LAUNCHER"
echo "  concurrency=$SMU3M_SOURCEANCHOR_SWEEP_CONCURRENCY"
echo "  batch_size=$SMU3M_BATCH_SIZE"
echo "  train_limit=$SMU3M_TRAIN_LIMIT"
echo "  hidden_dim=$SMU3M_DIFFUSION_HIDDEN_DIM"
echo "  depth=$SMU3M_DIFFUSION_DEPTH"
echo "  eval_limit=$SMU3M_SOURCEANCHOR_EVAL_LIMIT"
echo "  max_eval_per_property_count=$SMU3M_SOURCEANCHOR_MAX_EVAL_PER_PROPERTY_COUNT"
echo "  time=$SWEEP_TIME"
echo "  cpus=$SWEEP_CPUS"
echo "  mem_per_cpu=$SWEEP_MEM_PER_CPU"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"

SBATCH_ARGS=(
  --account="$SWEEP_ACCOUNT"
  --job-name="$SWEEP_JOB_NAME"
  --time="$SWEEP_TIME"
  --nodes=1
  --ntasks-per-node="$SWEEP_CPUS"
  --mem-per-cpu="$SWEEP_MEM_PER_CPU"
  --output="$SWEEP_LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$SWEEP_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$SWEEP_PARTITION")
fi

sweep_job_id=""
for gpu_request in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$gpu_request"
  if output="$(
    sbatch "${SBATCH_ARGS[@]}" \
      --gpus="$gpu_request" \
      --wrap="bash '$PROJECT_DIR/scripts/archive/legacy/run_unified_sourceanchor_sweep.sh'"
  )"; then
    echo "$output"
    sweep_job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done

if [[ -z "$sweep_job_id" ]]; then
  echo "ERROR: failed to submit source-anchor sweep." >&2
  exit 1
fi

echo
echo "Source-anchor sweep submitted."
echo "  job_id=$sweep_job_id"
echo "  log=$SWEEP_LOG_DIR/${SWEEP_JOB_NAME}-${sweep_job_id}.log"
echo "  task_file=$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT/tasks/sourceanchor_sweep.tasks"
echo "  manifest=$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT/tasks/sourceanchor_sweep_manifest.csv"
echo "  summary=$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT/sourceanchor_sweep_summary.md"
