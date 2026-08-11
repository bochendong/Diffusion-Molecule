#!/usr/bin/env bash
# Submit the CPU-only v5 DRD2 oracle completion on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_retrieved_delta_support_v5}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_UCA_CONTROLLER_ACCOUNT:-def-hup-ab_cpu}"
mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"

output="$(sbatch \
  --account="$ACCOUNT" \
  --job-name="${SUCC_UCA_JOB_NAME:-uca-delta-oracle-v5}" \
  --time="${SUCC_UCA_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_UCA_CPUS:-4}" \
  --mem="${SUCC_UCA_MEM:-8G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_retrieved_delta_oracle_repair_v5.sh'")"

job_id="$(printf '%s\n' "$output" | awk '{print $NF}')"
echo "$output"
echo "retrieved_delta_oracle_repair_job=$job_id"
