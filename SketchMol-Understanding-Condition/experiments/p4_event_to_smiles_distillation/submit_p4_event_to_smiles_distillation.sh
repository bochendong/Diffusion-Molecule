#!/usr/bin/env bash
# Submit the single-seed P4 distillation pilot on one H100 20 GB MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P4_SEED:-7}"
ACCOUNT="${P4_SLURM_ACCOUNT:-rrg-hup}"
GPU="${P4_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
TIME="${P4_SLURM_TIME:-04:00:00}"
MEM="${P4_SLURM_MEM:-64G}"
CPUS="${P4_SLURM_CPUS:-8}"
LOG_DIR="${P4_LOG_DIR:-$PROJECT_DIR/logs/p4_event_to_smiles_distillation_v1}"
MAIL_USER="${P4_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="p4-event-distill-s${SEED}" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p4-event-distill-s${SEED}-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p4_event_to_smiles_distillation.sh'")"
job_id="${submission%%;*}"
echo "P4 Event-to-SMILES distillation submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  time=$TIME"
echo "  log=$LOG_DIR/p4-event-distill-s${SEED}-${job_id}.log"
