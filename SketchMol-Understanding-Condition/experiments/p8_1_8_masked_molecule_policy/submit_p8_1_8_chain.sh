#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P818_SEED:-7}"
LOG_DIR="${P818_LOG_DIR:-$PROJECT_DIR/logs/p8_1_8_masked_molecule_policy}"
mkdir -p "$LOG_DIR"
r1="$(sbatch --parsable --account="${P818_SLURM_ACCOUNT:-rrg-hup}" \
  --job-name="p818-r1-s${SEED}" --time="${P818_R1_TIME:-00:15:00}" --mem=48G --cpus-per-task=8 \
  --gpus="${P818_SLURM_GPU:-h100:1}" --mail-user="${P818_MAIL_USER:-dongbochen1218@gmail.com}" \
  --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/p818-r1-s${SEED}-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p8_1_8_r1.sh'")"
r1_id="${r1%%;*}"
r2="$(sbatch --parsable --account="${P818_SLURM_ACCOUNT:-rrg-hup}" --dependency="afterok:${r1_id}" \
  --job-name="p818-r2-s${SEED}" --time="${P818_R2_TIME:-00:06:00}" --mem=48G --cpus-per-task=8 \
  --gpus="${P818_SLURM_GPU:-h100:1}" --mail-user="${P818_MAIL_USER:-dongbochen1218@gmail.com}" \
  --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/p818-r2-s${SEED}-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p8_1_8_r2.sh'")"
r2_id="${r2%%;*}"
echo "P8.1.8 two-round masked-molecule chain submitted"
echo "  r1_job_id=$r1_id"
echo "  r2_job_id=$r2_id"
echo "  r2_dependency=afterok:$r1_id"

