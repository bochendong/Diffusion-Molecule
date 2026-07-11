#!/usr/bin/env bash
# Submit CPU-only resume for phase1 Table1 benchmark metrics (reuse sample_outputs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_UNIFIED_PHASE1_TABLE1_RESUME_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_PHASE1_TABLE1_RESUME_SLURM_TIME:-02:00:00}"
MEM="${SUCC_UNIFIED_PHASE1_TABLE1_RESUME_SLURM_MEM:-16G}"
CPUS="${SUCC_UNIFIED_PHASE1_TABLE1_RESUME_SLURM_CPUS:-4}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_UNIFIED_PHASE1_TABLE1_RESUME_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
JOB_NAME="${SUCC_UNIFIED_PHASE1_TABLE1_RESUME_JOB_NAME:-succ-unified-phase1-table1-bench-resume}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

echo "Submitting unified phase1 Table1 benchmark resume (metrics only)"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  python=$SUCC_PYTHON_BIN"
echo "  output_root=${SUCC_UNIFIED_TABLE1_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_phase1_table1_direct_compat_v1}"
echo "  moledit_budgets=${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20,256}"
echo "  moledit_rdkit_modules=${SUCC_UNIFIED_MOLEDIT_RDKIT_MODULES:-gcc/12.3 rdkit/2024.09.6}"

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$SCRIPT_DIR/run_unified_phase1_table1_benchmark_resume.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit phase1 Table1 benchmark resume." >&2
  exit 1
fi
echo "unified_phase1_table1_benchmark_resume_job=$job_id"
