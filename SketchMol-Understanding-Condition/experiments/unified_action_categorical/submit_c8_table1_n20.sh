#!/usr/bin/env bash
# Submit C8: C7 graph scores + C5 seed 2005. CPU like C1/C6; C1 itself was 5m25s.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_DEVICE="${SUCC_DEVICE:-cpu}"
export SUCC_SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
ACCOUNT="${SUCC_C8_ACCOUNT:-def-hup-ab}"
TIME="${SUCC_C8_TIME:-00:20:00}"
MEM="${SUCC_C8_MEM:-32G}"
CPUS="${SUCC_C8_CPUS:-8}"
JOB_NAME="${SUCC_C8_JOB_NAME:-succ-c8-table1-n20}"
MAIL_USER="${SUCC_C8_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/joint_graph_fragment_categorical_c8}"

mkdir -p "$LOG_DIR"

job_id="$(sbatch --parsable \
  --account="$ACCOUNT" \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="$LOG_DIR/${JOB_NAME}-%j.log" \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_c8_table1_n20.sh'")"

echo "c8_table1_n20_job=$job_id"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
echo "summary=$PROJECT_DIR/outputs/joint_graph_fragment_categorical_c8/summary.json"
