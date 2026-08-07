#!/usr/bin/env bash
# Submit instruction-aligned GraphEditDSL v2 to one Nibi H100 20 GB MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${UMTP_GRAPH_ACTION_V2_SLURM_ACCOUNT:-def-hup-ab}"
GPU="${UMTP_GRAPH_ACTION_V2_SLURM_GPUS:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
MEM="${UMTP_GRAPH_ACTION_V2_SLURM_MEM:-48G}"
CPUS="${UMTP_GRAPH_ACTION_V2_SLURM_CPUS:-8}"
TIME="${UMTP_GRAPH_ACTION_V2_SLURM_TIME:-02:00:00}"
PARTITION="${UMTP_GRAPH_ACTION_V2_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
MAIL_USER="${UMTP_GRAPH_ACTION_V2_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/umtp_graph_action_instruction_v2}"
SEED="${UMTP_GRAPH_ACTION_V2_SEED:-7}"

mkdir -p "$LOG_DIR"

args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="umtp-action-v2-s${SEED}"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --gpus="$GPU"
  --output="$LOG_DIR/umtp-action-v2-s${SEED}-%j.log"
  --export=ALL
)
[[ -n "$PARTITION" ]] && args+=(--partition="$PARTITION")
if [[ -n "$MAIL_USER" ]]; then
  args+=(--mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL)
fi

submission="$(sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/run_umtp_graph_action_instruction_pilot.sh'")"
job_id="${submission%%;*}"

echo "Instruction-aligned GraphEditDSL v2 submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  time=$TIME"
echo "  oracle_gate=GSK3B fully-evaluable>=95%, strict-reachability>=5%"
echo "  mail=$MAIL_USER"
echo "  log=$LOG_DIR/umtp-action-v2-s${SEED}-${job_id}.log"
