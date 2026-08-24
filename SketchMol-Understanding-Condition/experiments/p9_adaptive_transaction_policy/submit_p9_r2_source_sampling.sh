#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P9_SEED:-7}"
LOG_DIR="${P9_R2_LOG_DIR:-$PROJECT_DIR/logs/p9_adaptive_transaction_policy_r2_source_sampling_v1}"
DEPENDENCY="${P9_R2_DEPENDENCY:-}"
mkdir -p "$LOG_DIR"
dependency_args=()
if [[ -n "$DEPENDENCY" ]]; then
  dependency_args+=(--dependency="$DEPENDENCY")
fi
submission="$(sbatch --parsable --account="${P9_SLURM_ACCOUNT:-rrg-hup}" \
  --job-name="p9-r2-source-s${SEED}" --time="${P9_R2_SLURM_TIME:-00:06:00}" \
  --mem="${P9_R2_SLURM_MEM:-48G}" --cpus-per-task="${P9_R2_SLURM_CPUS:-8}" \
  --gpus="${P9_R2_SLURM_GPU:-h100:1}" "${dependency_args[@]}" \
  --mail-user="${P9_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p9-r2-source-s${SEED}-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_p9_r2_source_sampling.sh'")"
job_id="${submission%%;*}"
echo "P9-R2 direct source-only sampling submitted"
echo "  job_id=$job_id"
echo "  dependency=${DEPENDENCY:-none}"
echo "  log=$LOG_DIR/p9-r2-source-s${SEED}-${job_id}.log"
