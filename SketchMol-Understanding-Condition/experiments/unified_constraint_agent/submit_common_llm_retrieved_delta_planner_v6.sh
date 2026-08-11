#!/usr/bin/env bash
# Submit the one-seed v6 common-LLM RetrievedDelta planner signal on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_DELTA_PLANNER_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_planner_v6}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_retrieved_delta_planner_v6}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_UCA_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
SEED="${SUCC_UCA_SEED:-1709}"

mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_DELTA_PLANNER_ROOT="$RUN_ROOT"

echo "Submitting common-LLM RetrievedDelta planner v6"
echo "  seed=$SEED"
echo "  final_oracle_candidate_budget=20"
echo "  evaluation_target_access=false"
echo "  planner_candidate_limit=96"
echo "  gpu=$GPU_REQUEST"
echo "  expected_runtime=30-60 minutes"

output="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="${SUCC_UCA_JOB_NAME:-uca-delta-plan-v6-s${SEED}}" \
  --time="${SUCC_UCA_TIME:-02:00:00}" \
  --cpus-per-task="${SUCC_UCA_CPUS:-4}" \
  --mem="${SUCC_UCA_MEM:-32G}" \
  --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_common_llm_retrieved_delta_planner_v6.sh'")"

job_id="$(printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1)"
[[ -n "$job_id" ]] || { echo "ERROR: could not parse Slurm job id from: $output" >&2; exit 2; }
echo "$output"
echo "retrieved_delta_planner_job=$job_id"
echo "output=$RUN_ROOT"
