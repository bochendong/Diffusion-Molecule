#!/usr/bin/env bash
# Submit the MolEditRL paper-faithful Table1 materialization/evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

DRY_RUN="${DRY_RUN:-${MOLEDITRL_DRY_RUN:-0}}"
JOB_NAME="${MOLEDITRL_SLURM_JOB_NAME:-moleditrl-table1-paper}"
ACCOUNT="${MOLEDITRL_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${MOLEDITRL_SLURM_TIME:-02:00:00}"
CPUS="${MOLEDITRL_SLURM_CPUS:-2}"
MEM="${MOLEDITRL_SLURM_MEM:-16G}"
PARTITION="${MOLEDITRL_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
DEPENDENCY="${MOLEDITRL_SLURM_DEPENDENCY:-${SUCC_SLURM_DEPENDENCY:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"

mkdir -p "$LOG_DIR"

echo "Submitting MolEditRL paper-faithful Table1"
echo "  dry_run=$DRY_RUN"
echo "  job_name=$JOB_NAME"
echo "  output_dir=${MOLEDITRL_OUTPUT_DIR:-Research/Molecule Generation/OfficialBaselines/MolEditRL/results/table1_paper_faithful}"
echo "  predictions_csv=${MOLEDITRL_PREDICTIONS_CSV:-none}"
echo "  dependency=${DEPENDENCY:-none}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "would run: bash '$PROJECT_DIR/scripts/run_moleditrl_table1_paper_faithful.sh'"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

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
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_moleditrl_table1_paper_faithful.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit MolEditRL paper-faithful Table1 job." >&2
  exit 1
fi

echo "moleditrl_table1_paper_job=$job_id"
