#!/usr/bin/env bash
# Submit the shortest leakage-safe RetrievedDeltaEdit support signal on Nibi.

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
RUN_ROOT="${SUCC_UCA_RETRIEVED_DELTA_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_support_v5}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_retrieved_delta_support_v5}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_UCA_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"

mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_RETRIEVED_DELTA_ROOT="$RUN_ROOT"

echo "Submitting RetrievedDeltaEdit support gate"
echo "  protocol=train-only matched-pair delta retrieval + final oracle n=20"
echo "  audit_target_access=false"
echo "  gpu=$GPU_REQUEST"
echo "  time=${SUCC_UCA_TIME:-01:00:00}"
echo "  memory=${SUCC_UCA_MEM:-24G}"
echo "  expected_runtime=10-25 minutes"

output="$(sbatch \
  --account="$ACCOUNT" \
  --job-name="${SUCC_UCA_JOB_NAME:-uca-delta-support-v5}" \
  --time="${SUCC_UCA_TIME:-01:00:00}" \
  --cpus-per-task="${SUCC_UCA_CPUS:-4}" \
  --mem="${SUCC_UCA_MEM:-24G}" \
  --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_common_agent_retrieved_delta_support_v5.sh'")"

job_id="$(printf '%s\n' "$output" | awk '{print $NF}')"
echo "$output"
echo "retrieved_delta_support_job=$job_id"
echo "output=$RUN_ROOT"
