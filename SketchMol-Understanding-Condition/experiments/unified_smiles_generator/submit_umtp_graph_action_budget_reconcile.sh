#!/usr/bin/env bash
# Re-evaluate the saved protected GraphEditDSL candidate pool at paper budgets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${UMTP_GRAPH_ACTION_RECONCILE_SLURM_ACCOUNT:-rrg-hup}"
MEM="${UMTP_GRAPH_ACTION_RECONCILE_SLURM_MEM:-32G}"
CPUS="${UMTP_GRAPH_ACTION_RECONCILE_SLURM_CPUS:-8}"
TIME="${UMTP_GRAPH_ACTION_RECONCILE_SLURM_TIME:-01:00:00}"
PARTITION="${UMTP_GRAPH_ACTION_RECONCILE_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
MAIL_USER="${UMTP_GRAPH_ACTION_RECONCILE_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/umtp_graph_action_full_eval_v1}"
SEED="${UMTP_GRAPH_ACTION_SEED:-7}"
BUDGETS="${UMTP_GRAPH_ACTION_FULL_BUDGETS:-1,8,20,64,256}"

mkdir -p "$LOG_DIR"

args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="umtp-action-k20-s${SEED}"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/umtp-action-k20-s${SEED}-%j.log"
  --export="ALL,UMTP_GRAPH_ACTION_RUN_RANK=0,UMTP_GRAPH_ACTION_FORCE=1,UMTP_GRAPH_ACTION_FULL_BUDGETS=$BUDGETS"
)
[[ -n "$PARTITION" ]] && args+=(--partition="$PARTITION")
if [[ -n "$MAIL_USER" ]]; then
  args+=(--mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL)
fi

submission="$(sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/run_umtp_graph_action_full_eval.sh'")"
job_id="${submission%%;*}"

echo "Protected GraphEditDSL budget reconciliation submitted"
echo "  job_id=$job_id"
echo "  budgets=$BUDGETS"
echo "  candidate_generation=reused"
echo "  gpu=none"
echo "  time=$TIME"
echo "  mail=$MAIL_USER"
echo "  log=$LOG_DIR/umtp-action-k20-s${SEED}-${job_id}.log"
