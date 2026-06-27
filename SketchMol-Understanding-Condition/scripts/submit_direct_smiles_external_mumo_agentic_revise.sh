#!/usr/bin/env bash
# Submit MuMO source-conditioned agentic revise benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_EXTERNAL_AGENTIC_SOURCE_FILE="${SUCC_EXTERNAL_AGENTIC_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
export SUCC_EXTERNAL_AGENTIC_OUTPUT_DIR="${SUCC_EXTERNAL_AGENTIC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1}"
export SUCC_EXTERNAL_AGENTIC_SUITE="${SUCC_EXTERNAL_AGENTIC_SUITE:-mumo}"
export SUCC_EXTERNAL_AGENTIC_TASK_SPLIT="${SUCC_EXTERNAL_AGENTIC_TASK_SPLIT:-all}"
export SUCC_EXTERNAL_AGENTIC_MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_AGENTIC_MAX_ROWS_PER_TASK:-200}"
export SUCC_EXTERNAL_AGENTIC_CONDITION_MIXING_MODE="${SUCC_EXTERNAL_AGENTIC_CONDITION_MIXING_MODE:-append_source_property_program}"
export SUCC_EXTERNAL_AGENTIC_STEPS="${SUCC_EXTERNAL_AGENTIC_STEPS:-2}"
export SUCC_EXTERNAL_AGENTIC_SLURM_JOB_NAME="${SUCC_EXTERNAL_AGENTIC_SLURM_JOB_NAME:-succ-external-mumo-agentic-revise}"

ACCOUNT="${SUCC_EXTERNAL_AGENTIC_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_EXTERNAL_AGENTIC_SLURM_TIME:-${SUCC_SLURM_TIME:-8:00:00}}"
MEM="${SUCC_EXTERNAL_AGENTIC_SLURM_MEM:-${SUCC_SLURM_MEM:-96G}}"
CPUS="${SUCC_EXTERNAL_AGENTIC_SLURM_CPUS:-${SUCC_SLURM_CPUS:-8}}"
JOB_NAME="$SUCC_EXTERNAL_AGENTIC_SLURM_JOB_NAME"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_EXTERNAL_AGENTIC_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
GPU_PROFILE="${SUCC_EXTERNAL_AGENTIC_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_40gb_mig}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "$SUCC_EXTERNAL_AGENTIC_SOURCE_FILE" || ! -f "$SUCC_EXTERNAL_AGENTIC_SOURCE_FILE" ]]; then
  echo "ERROR: source file not found: $SUCC_EXTERNAL_AGENTIC_SOURCE_FILE" >&2
  exit 2
fi

if [[ -n "${SUCC_EXTERNAL_AGENTIC_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_EXTERNAL_AGENTIC_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "none" || "$GPU_PROFILE" == "0" ]]; then
  GPU_CANDIDATES=("")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting MuMO agentic revise benchmark"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  source_file=$SUCC_EXTERNAL_AGENTIC_SOURCE_FILE"
echo "  output_dir=$SUCC_EXTERNAL_AGENTIC_OUTPUT_DIR"
echo "  task_split=$SUCC_EXTERNAL_AGENTIC_TASK_SPLIT"
echo "  max_rows_per_task=$SUCC_EXTERNAL_AGENTIC_MAX_ROWS_PER_TASK"
echo "  agentic_steps=$SUCC_EXTERNAL_AGENTIC_STEPS"
echo "  gpu_candidates=${GPU_CANDIDATES[*]:-none}"

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
  if [[ -n "$GPU_REQUEST" ]]; then
    echo "Trying sbatch with --gpus=$GPU_REQUEST"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_external_mumo_agentic_revise.sh'")"; then
      continue
    fi
  else
    echo "Trying sbatch without GPU request"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_external_mumo_agentic_revise.sh'")"; then
      continue
    fi
  fi
  echo "$output"
  job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -n "$job_id" ]]; then
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit MuMO agentic revise benchmark." >&2
  exit 1
fi

echo "external_mumo_agentic_revise_job=$job_id"
