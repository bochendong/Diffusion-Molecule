#!/usr/bin/env bash
# Submit a packed source-aware parameter sweep as one Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SMU3M_SWEEP_OUTPUT_ROOT="${SMU3M_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2}"
export SMU3M_SWEEP_LAUNCHER="${SMU3M_SWEEP_LAUNCHER:-glost}"
export SMU3M_SWEEP_CONCURRENCY="${SMU3M_SWEEP_CONCURRENCY:-1}"
export SMU3M_SWEEP_RESUME="${SMU3M_SWEEP_RESUME:-0}"
export SMU3M_SWEEP_PRESET="${SMU3M_SWEEP_PRESET:-extended}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

# Keep each packed subtask short and comparable by default. Override these when
# you want the same sweep over the full training/eval profile.
export SMU3M_RESOURCE_PROFILE="${SMU3M_RESOURCE_PROFILE:-economy}"
export SMU3M_GPU_PROFILE="${SMU3M_GPU_PROFILE:-h100_20gb_mig}"
export SMU3M_DESCRIPTION_LIMIT="${SMU3M_DESCRIPTION_LIMIT:-5000}"
export SMU3M_EDIT_LIMIT="${SMU3M_EDIT_LIMIT:-50000}"
export SMU3M_TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
export SMU3M_EPOCHS="${SMU3M_EPOCHS:-5}"
export SMU3M_EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-1000}"
export SMU3M_EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
export SMU3M_REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
export SMU3M_RESUME="${SMU3M_RESUME:-0}"
export SMMED_SUBMIT_DATASET_BUILD="${SMMED_SUBMIT_DATASET_BUILD:-0}"

SWEEP_ACCOUNT="${SMU3M_SWEEP_SLURM_ACCOUNT:-${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
SWEEP_TIME="${SMU3M_SWEEP_SLURM_TIME:-01:00:00}"
SWEEP_CPUS="${SMU3M_SWEEP_SLURM_CPUS:-$SMU3M_SWEEP_CONCURRENCY}"
SWEEP_MEM_PER_CPU="${SMU3M_SWEEP_SLURM_MEM_PER_CPU:-6G}"
SWEEP_JOB_NAME="${SMU3M_SWEEP_SLURM_JOB_NAME:-smu3m-src-sweep}"
SWEEP_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
SWEEP_PARTITION="${SMU3M_SWEEP_SLURM_PARTITION:-${SMU3M_SLURM_PARTITION:-}}"

if (( SMU3M_SWEEP_CONCURRENCY <= 0 )); then
  echo "ERROR: SMU3M_SWEEP_CONCURRENCY must be positive, got $SMU3M_SWEEP_CONCURRENCY" >&2
  exit 2
fi
if (( SWEEP_CPUS < SMU3M_SWEEP_CONCURRENCY )); then
  echo "ERROR: SMU3M_SWEEP_SLURM_CPUS=$SWEEP_CPUS is less than SMU3M_SWEEP_CONCURRENCY=$SMU3M_SWEEP_CONCURRENCY" >&2
  exit 2
fi
if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SMU3M_SWEEP_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SWEEP_SLURM_GPUS")
elif [[ -n "${SMU3M_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SMU3M_SLURM_GPUS")
elif [[ "$SMU3M_GPU_PROFILE" == "none" || "$SMU3M_REQUIRE_CUDA" == "0" ]]; then
  GPU_CANDIDATES=("")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$SMU3M_GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
else
  GPU_CANDIDATES=("$SMU3M_GPU_PROFILE")
fi

mkdir -p "$SWEEP_LOG_DIR"

echo "Submitting packed Unified 3M source-aware sweep"
echo "  output_root=$SMU3M_SWEEP_OUTPUT_ROOT"
echo "  launcher=$SMU3M_SWEEP_LAUNCHER"
echo "  preset=$SMU3M_SWEEP_PRESET"
echo "  concurrency=$SMU3M_SWEEP_CONCURRENCY"
echo "  sweep_time=$SWEEP_TIME"
echo "  sweep_cpus=$SWEEP_CPUS"
echo "  mem_per_cpu=$SWEEP_MEM_PER_CPU"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  gpu_candidates=${GPU_CANDIDATES[*]:-(none)}"

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
  if [[ -n "$gpu_request" ]]; then
    echo "Trying sbatch with --gpus=$gpu_request"
    if output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$gpu_request" --wrap="bash '$PROJECT_DIR/scripts/run_unified_sourceaware_sweep.sh'")"; then
      echo "$output"
      sweep_job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
      break
    fi
  else
    echo "Trying sbatch without GPU request"
    if output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_unified_sourceaware_sweep.sh'")"; then
      echo "$output"
      sweep_job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
      break
    fi
  fi
done

if [[ -z "$sweep_job_id" ]]; then
  echo "ERROR: failed to submit source-aware sweep." >&2
  exit 1
fi

echo
echo "Source-aware sweep submitted."
echo "  job_id=$sweep_job_id"
echo "  log=$SWEEP_LOG_DIR/${SWEEP_JOB_NAME}-${sweep_job_id}.log"
echo "  task_file=$SMU3M_SWEEP_OUTPUT_ROOT/tasks/sourceaware_sweep.tasks"
echo "  summary=$SMU3M_SWEEP_OUTPUT_ROOT/sweep_summary.md"
