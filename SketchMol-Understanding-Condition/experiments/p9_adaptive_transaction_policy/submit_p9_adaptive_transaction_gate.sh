#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P9_SEED:-7}"
LOG_DIR="${P9_LOG_DIR:-$PROJECT_DIR/logs/p9_adaptive_transaction_policy_v1}"
mkdir -p "$LOG_DIR"
submission="$(sbatch --parsable --account="${P9_SLURM_ACCOUNT:-rrg-hup}" \
  --job-name="p9-adapt-txn-s${SEED}" --time="${P9_SLURM_TIME:-00:30:00}" \
  --mem="${P9_SLURM_MEM:-64G}" --cpus-per-task="${P9_SLURM_CPUS:-8}" \
  --gpus="${P9_SLURM_GPU:-h100:1}" --mail-user="${P9_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" \
  --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/p9-adapt-txn-s${SEED}-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_p9_adaptive_transaction_gate.sh'")"
job_id="${submission%%;*}"
echo "P9 adaptive transaction gate submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/p9-adapt-txn-s${SEED}-${job_id}.log"
