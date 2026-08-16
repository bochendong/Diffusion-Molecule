#!/usr/bin/env bash
# Submit one bounded CPU-only contextual transition operator signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_CONTEXTUAL_OPERATOR_SEED:-2019}"
LOG_DIR="${SUCC_CONTEXTUAL_OPERATOR_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/contextual_source_conditioned_transition_operator_v1}"
MAIL_USER="${SUCC_CONTEXTUAL_OPERATOR_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-context-op-v1" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_CONTEXTUAL_OPERATOR_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_CONTEXTUAL_OPERATOR_CPUS:-4}" \
  --mem="${SUCC_CONTEXTUAL_OPERATOR_MEM:-8G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-context-op-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_contextual_source_conditioned_transition_operator.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "contextual_operator_job=$job_id"
echo "log=$LOG_DIR/uca-context-op-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/contextual_source_conditioned_transition_operator_v1/seed_${SEED}/summary.json"
