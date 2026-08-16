#!/usr/bin/env bash
# Submit one bounded CPU-only property-aligned balanced transport signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_BALANCED_TXN_SEED:-2011}"
LOG_DIR="${SUCC_BALANCED_TXN_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/property_aligned_balanced_transaction_transport_v1}"
MAIL_USER="${SUCC_BALANCED_TXN_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-prop-bal-txn-v1" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_BALANCED_TXN_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_BALANCED_TXN_CPUS:-4}" \
  --mem="${SUCC_BALANCED_TXN_MEM:-8G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-prop-bal-txn-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_property_aligned_balanced_transaction_transport.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "property_aligned_balanced_job=$job_id"
echo "log=$LOG_DIR/uca-prop-bal-txn-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/property_aligned_balanced_transaction_transport_v1/seed_${SEED}/summary.json"
