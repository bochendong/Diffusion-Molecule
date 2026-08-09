#!/usr/bin/env bash
# Submit the CPU-only fixed-budget teacher audit on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

CONFIG_JSON="${SUCC_UCA_CONFIG_JSON:?Set SUCC_UCA_CONFIG_JSON to the run manifest}"
OUTPUT_DIR="${SUCC_UCA_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_constraint_agent_teacher_audit_v1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-SketchMol-Understanding-Condition/logs/unified_constraint_agent_teacher_audit_v1}"
PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
export SUCC_UCA_CONFIG_JSON="$CONFIG_JSON"
export SUCC_UCA_OUTPUT_DIR="$OUTPUT_DIR"
export SUCC_UCA_PYTHON_BIN="$PYTHON_BIN"

output="$(sbatch \
  --job-name="${SUCC_UCA_JOB_NAME:-uca-teacher-audit}" \
  --time="${SUCC_UCA_TIME:-00:30:00}" \
  --cpus-per-task=4 \
  --mem=16G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_teacher_pool_audit.sh'")"

job_id="$(printf '%s\n' "$output" | awk '{print $NF}')"
echo "Unified constraint teacher audit submitted"
echo "  job_id=$job_id"
echo "  config=$CONFIG_JSON"
echo "  output=$OUTPUT_DIR"
