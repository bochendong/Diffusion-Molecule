#!/usr/bin/env bash
# Submit CPU any@20 re-eval of existing Table1 group-RL candidates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_TABLE1_NORANK_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab}}"
TIME="${SUCC_TABLE1_NORANK_TIME:-01:00:00}"
MEM="${SUCC_TABLE1_NORANK_MEM:-32G}"
CPUS="${SUCC_TABLE1_NORANK_CPUS:-8}"
JOB_NAME="${SUCC_TABLE1_NORANK_JOB_NAME:-succ-table1-any20}"
MAIL_USER="${SUCC_TABLE1_NORANK_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/direct_smiles_table1_n20_norank}"
PARTITION="${SUCC_TABLE1_NORANK_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

mkdir -p "$LOG_DIR"

args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --mail-user="$MAIL_USER"
  --mail-type=END,FAIL
  --export=ALL
)
[[ -n "$PARTITION" ]] && args+=(--partition="$PARTITION")

job_id="$(sbatch "${args[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_table1_n20_norank_reeval.sh'")"
echo "table1_n20_norank_job=$job_id"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
