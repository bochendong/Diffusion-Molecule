#!/usr/bin/env bash
# Submit the CPU-only common-LLM train-data preparation job on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_common_llm_pilot_v1}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

output="$(sbatch \
  --job-name="${SUCC_UCA_DATA_JOB_NAME:-uca-llm-data}" \
  --time="${SUCC_UCA_DATA_TIME:-04:00:00}" \
  --cpus-per-task="${SUCC_UCA_DATA_CPUS:-8}" \
  --mem="${SUCC_UCA_DATA_MEM:-64G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_common_llm_data_prep.sh'")"

job_id="$(printf '%s\n' "$output" | awk '{print $NF}')"
echo "Common-LLM data preparation submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/uca-llm-data-$job_id.log"
