#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${P812_SEED:-7}"
LOG_DIR="${P812_R2_LOG_DIR:-$PROJECT_DIR/logs/p8_1_2_unified_transduction_raw_r2_p1_v1}"
OUTPUT_ROOT="${P812_R2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p8_1_2_unified_transduction_raw_r2_p1_v1/seed_${SEED}}"
P1_CHECKPOINT="${P812_R2_BASE_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
mkdir -p "$LOG_DIR"
DEPENDENCY_ARGS=()
if [[ -n "${P812_DEPENDENCY:-}" ]]; then
  DEPENDENCY_ARGS+=(--dependency="${P812_DEPENDENCY}")
fi
submission="$(sbatch --parsable --account="${P812_SLURM_ACCOUNT:-def-hup-ab}" \
  --job-name="p812-r2-p1-s${SEED}" --time="${P812_SLURM_TIME:-00:15:00}" \
  --mem="${P812_SLURM_MEM:-64G}" --cpus-per-task="${P812_SLURM_CPUS:-8}" \
  --gpus="${P812_SLURM_GPU:-h100:1}" "${DEPENDENCY_ARGS[@]}" \
  --mail-user="${P812_SLURM_MAIL_USER:-dongbochen1218@gmail.com}" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/p812-r2-p1-s${SEED}-%j.log" --export=ALL \
  --wrap="P812_BASE_CHECKPOINT='$P1_CHECKPOINT' P812_SOURCE_AWARE_WARMSTART=1 P812_OUTPUT_ROOT='$OUTPUT_ROOT' bash '$SCRIPT_DIR/run_p8_1_2_raw_gate.sh'")"
job_id="${submission%%;*}"
echo "P8.1.2 raw R2 P1-warmstart gate submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/p812-r2-p1-s${SEED}-${job_id}.log"
