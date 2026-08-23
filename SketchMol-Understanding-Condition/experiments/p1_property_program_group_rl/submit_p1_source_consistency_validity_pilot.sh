#!/usr/bin/env bash
# Submit the validation-only consistency/validity pilot to one short H100 job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${P1_CONSISTENCY_SLURM_ACCOUNT:-rrg-hup}"
GPU="${P1_CONSISTENCY_SLURM_GPUS:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
MEM="${P1_CONSISTENCY_SLURM_MEM:-64G}"
CPUS="${P1_CONSISTENCY_SLURM_CPUS:-8}"
TIME="${P1_CONSISTENCY_SLURM_TIME:-02:00:00}"
PARTITION="${P1_CONSISTENCY_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
SEED="${P1_CONSISTENCY_SEED:-7}"
LOG_DIR="${P1_CONSISTENCY_LOG_DIR:-$PROJECT_DIR/logs/p1_source_consistency_validity_v1}"

mkdir -p "$LOG_DIR"
args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="p1-consistency-s${SEED}"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --gpus="$GPU"
  --output="$LOG_DIR/p1-consistency-s${SEED}-%j.log"
  --export=ALL
)
[[ -n "$PARTITION" ]] && args+=(--partition="$PARTITION")

submission="$(sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/run_p1_source_consistency_validity_pilot.sh'")"
job_id="${submission%%;*}"

echo "P1 source-consistency + validity pilot submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  time=$TIME"
echo "  log=$LOG_DIR/p1-consistency-s${SEED}-${job_id}.log"
