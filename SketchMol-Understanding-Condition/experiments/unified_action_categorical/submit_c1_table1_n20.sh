#!/usr/bin/env bash
# Submit C1 frozen graph+fragment n=20 Table1 eval. CPU is enough; B31 family is CPU-native.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_DEVICE="${SUCC_DEVICE:-cpu}"
export SUCC_SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
ACCOUNT="${SUCC_C1_ACCOUNT:-def-hup-ab}"
TIME="${SUCC_C1_TIME:-01:00:00}"
MEM="${SUCC_C1_MEM:-32G}"
CPUS="${SUCC_C1_CPUS:-8}"
JOB_NAME="${SUCC_C1_JOB_NAME:-succ-c1-table1-n20}"
MAIL_USER="${SUCC_C1_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/joint_graph_fragment_categorical_c1}"

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
  --wrap="bash '$SCRIPT_DIR/run_c1_table1_n20.sh'")"

echo "c1_table1_n20_job=$job_id"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
echo "summary=$PROJECT_DIR/outputs/joint_graph_fragment_categorical_c1/summary.json"
