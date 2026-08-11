#!/usr/bin/env bash
# Submit the CPU-only composed RetrievedDelta support gate on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_COMPOSED_DELTA_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_composed_delta_support_v7}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_composed_delta_support_v7}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_UCA_ACCOUNT:-def-hup-ab_cpu}"

mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_COMPOSED_DELTA_ROOT="$RUN_ROOT"

echo "Submitting composed RetrievedDelta support gate"
echo "  accelerator=none"
echo "  immutable_anchor_budget=20"
echo "  max_delta_steps=2"
echo "  diagnostic_candidate_limit=96"
echo "  evaluation_target_access=false"
echo "  expected_runtime=8-20 minutes"

output="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="${SUCC_UCA_JOB_NAME:-uca-compose-delta-v7}" \
  --time="${SUCC_UCA_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_UCA_CPUS:-4}" \
  --mem="${SUCC_UCA_MEM:-12G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_composed_retrieved_delta_support_v7.sh'")"

job_id="$(printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1)"
[[ -n "$job_id" ]] || { echo "ERROR: could not parse Slurm job id from: $output" >&2; exit 2; }
echo "$output"
echo "composed_delta_support_job=$job_id"
echo "output=$RUN_ROOT"
