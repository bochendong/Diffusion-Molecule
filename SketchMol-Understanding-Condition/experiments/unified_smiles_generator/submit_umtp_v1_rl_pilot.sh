#!/usr/bin/env bash
# Submit one fast 20 GB MIG UMTP short-RL go/no-go job on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${UMTP_RL_PILOT_SLURM_ACCOUNT:-rrg-hup}"
GPU="${UMTP_RL_PILOT_SLURM_GPUS:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
MEM="${UMTP_RL_PILOT_SLURM_MEM:-48G}"
CPUS="${UMTP_RL_PILOT_SLURM_CPUS:-8}"
TIME="${UMTP_RL_PILOT_SLURM_TIME:-04:00:00}"
PARTITION="${UMTP_RL_PILOT_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
MAIL_USER="${UMTP_RL_PILOT_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/umtp_rl_pilot_v1}"
SEED="${UMTP_RL_PILOT_SEED:-7}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_v1}"
BASE_CHECKPOINT="${UMTP_RL_PILOT_BASE_CHECKPOINT:-$POLICY_ROOT/seed_${SEED}/policy/unified_smiles_generator.pt}"

[[ -f "$BASE_CHECKPOINT" ]] || { echo "ERROR: missing UMTP checkpoint: $BASE_CHECKPOINT" >&2; exit 2; }
mkdir -p "$LOG_DIR"

args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="umtp-rl-pilot-s${SEED}"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --gpus="$GPU"
  --output="$LOG_DIR/umtp-rl-pilot-s${SEED}-%j.log"
  --export=ALL
)
[[ -n "$PARTITION" ]] && args+=(--partition="$PARTITION")
if [[ -n "$MAIL_USER" ]]; then
  args+=(--mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL)
fi

submission="$(sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/run_umtp_v1_rl_pilot.sh'")"
job_id="${submission%%;*}"

echo "UMTP short RL pilot submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  time=$TIME"
echo "  checkpoint=$BASE_CHECKPOINT"
echo "  log=$LOG_DIR/umtp-rl-pilot-s${SEED}-${job_id}.log"
