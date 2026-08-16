#!/usr/bin/env bash
# Submit one bounded CPU signal for the atomic closed-transaction decoder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_ATOMIC_TRANSACTION_SEED:-2005}"
LOG_DIR="${SUCC_ATOMIC_TRANSACTION_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/atomic_closed_transaction_latent_decoder_v1}"
MAIL_USER="${SUCC_ATOMIC_TRANSACTION_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-atomic-txn-v1" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_ATOMIC_TRANSACTION_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_ATOMIC_TRANSACTION_CPUS:-8}" \
  --mem="${SUCC_ATOMIC_TRANSACTION_MEM:-24G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-atomic-txn-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_atomic_closed_transaction_latent_decoder.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "atomic_transaction_job=$job_id"
echo "log=$LOG_DIR/uca-atomic-txn-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/atomic_closed_transaction_latent_decoder_v1/seed_${SEED}/summary.json"
