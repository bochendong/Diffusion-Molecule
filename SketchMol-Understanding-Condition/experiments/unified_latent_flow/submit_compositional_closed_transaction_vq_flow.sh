#!/usr/bin/env bash
# Submit one bounded CPU-only compositional closed-transaction VQ signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_COMPOSITIONAL_VQ_SEED:-2007}"
LOG_DIR="${SUCC_COMPOSITIONAL_VQ_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/compositional_closed_transaction_vq_flow_v1}"
MAIL_USER="${SUCC_COMPOSITIONAL_VQ_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-comptxn-vq-v1" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_COMPOSITIONAL_VQ_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_COMPOSITIONAL_VQ_CPUS:-4}" \
  --mem="${SUCC_COMPOSITIONAL_VQ_MEM:-8G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-comptxn-vq-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_compositional_closed_transaction_vq_flow.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "compositional_vq_job=$job_id"
echo "log=$LOG_DIR/uca-comptxn-vq-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/compositional_closed_transaction_vq_flow_v1/seed_${SEED}/summary.json"
