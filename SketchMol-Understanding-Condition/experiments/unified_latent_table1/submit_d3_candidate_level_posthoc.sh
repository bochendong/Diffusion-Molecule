#!/usr/bin/env bash
# Submit the CPU-only D3 candidate-level Table1 evaluation on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

ACCOUNT="${SUCC_D3_POSTHOC_ACCOUNT:-def-hup-ab}"
TIME="${SUCC_D3_POSTHOC_TIME:-00:30:00}"
MEM="${SUCC_D3_POSTHOC_MEM:-16G}"
CPUS="${SUCC_D3_POSTHOC_CPUS:-4}"
JOB_NAME="${SUCC_D3_POSTHOC_JOB_NAME:-d3-candidate-metrics}"
MAIL_USER="${SUCC_D3_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/d3_candidate_level_posthoc}"
mkdir -p "$LOG_DIR"

submission="$(sbatch --parsable \
  --account="$ACCOUNT" \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="$LOG_DIR/${JOB_NAME}-%j.log" \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_d3_candidate_level_posthoc.sh'")"
job_id="${submission%%;*}"

echo "d3_candidate_level_posthoc_job=$job_id"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
