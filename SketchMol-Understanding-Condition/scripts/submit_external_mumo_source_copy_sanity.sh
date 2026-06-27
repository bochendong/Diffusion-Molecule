#!/usr/bin/env bash
# Submit MuMO source-copy sanity evaluation.

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
export SUCC_EXTERNAL_SOURCE_COPY_SOURCE_FILE="${SUCC_EXTERNAL_SOURCE_COPY_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
export SUCC_EXTERNAL_SOURCE_COPY_SUITE="${SUCC_EXTERNAL_SOURCE_COPY_SUITE:-mumo}"
export SUCC_EXTERNAL_SOURCE_COPY_TASK_SPLIT="${SUCC_EXTERNAL_SOURCE_COPY_TASK_SPLIT:-all}"
export SUCC_EXTERNAL_SOURCE_COPY_INPUT_SPLIT="${SUCC_EXTERNAL_SOURCE_COPY_INPUT_SPLIT:-all}"
export SUCC_EXTERNAL_SOURCE_COPY_MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_SOURCE_COPY_MAX_ROWS_PER_TASK:-200}"
export SUCC_EXTERNAL_SOURCE_COPY_OUTPUT_DIR="${SUCC_EXTERNAL_SOURCE_COPY_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_source_copy_sanity}"

ACCOUNT="${SUCC_EXTERNAL_SOURCE_COPY_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_EXTERNAL_SOURCE_COPY_SLURM_TIME:-${SUCC_SLURM_TIME:-01:00:00}}"
MEM="${SUCC_EXTERNAL_SOURCE_COPY_SLURM_MEM:-${SUCC_SLURM_MEM:-16G}}"
CPUS="${SUCC_EXTERNAL_SOURCE_COPY_SLURM_CPUS:-${SUCC_SLURM_CPUS:-2}}"
JOB_NAME="${SUCC_EXTERNAL_SOURCE_COPY_SLURM_JOB_NAME:-succ-external-mumo-source-copy}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_EXTERNAL_SOURCE_COPY_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

echo "Submitting MuMO source-copy sanity"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  source_file=$SUCC_EXTERNAL_SOURCE_COPY_SOURCE_FILE"
echo "  output_dir=$SUCC_EXTERNAL_SOURCE_COPY_OUTPUT_DIR"
echo "  suite=$SUCC_EXTERNAL_SOURCE_COPY_SUITE"
echo "  task_split=$SUCC_EXTERNAL_SOURCE_COPY_TASK_SPLIT"
echo "  max_rows_per_task=$SUCC_EXTERNAL_SOURCE_COPY_MAX_ROWS_PER_TASK"

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

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_multiproperty_source_copy_sanity.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit MuMO source-copy sanity." >&2
  exit 1
fi

echo "external_mumo_source_copy_job=$job_id"
