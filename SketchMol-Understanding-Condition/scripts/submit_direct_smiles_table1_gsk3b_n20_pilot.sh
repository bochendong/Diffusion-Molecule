#!/usr/bin/env bash
# Submit the fast GSK3B n=20 no-rank group-RL pilot. Prefer a full H100.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_TABLE1_GSK3B_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab}}"
TIME="${SUCC_TABLE1_GSK3B_TIME:-03:00:00}"
MEM="${SUCC_TABLE1_GSK3B_MEM:-64G}"
CPUS="${SUCC_TABLE1_GSK3B_CPUS:-8}"
JOB_NAME="${SUCC_TABLE1_GSK3B_JOB_NAME:-succ-gsk3b-n20}"
MAIL_USER="${SUCC_TABLE1_GSK3B_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs/direct_smiles_table1_gsk3b_n20_pilot}"
PARTITION="${SUCC_TABLE1_GSK3B_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

if [[ -n "${SUCC_TABLE1_GSK3B_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_TABLE1_GSK3B_GPUS")
else
  # Fastest first: full H100, then 40GB MIG, then 20GB MIG like today's jobs.
  GPU_CANDIDATES=(
    "h100:1"
    "nvidia_h100_80gb_hbm3_3g.40gb:1"
    "nvidia_h100_80gb_hbm3_2g.20gb:1"
  )
fi

mkdir -p "$LOG_DIR"

echo "Submitting GSK3B n=20 no-rank group-RL pilot"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"

SBATCH_ARGS=(
  --parsable
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --mail-user="$MAIL_USER"
  --mail-type=END,FAIL
  --export=ALL
)
[[ -n "$PARTITION" ]] && SBATCH_ARGS+=(--partition="$PARTITION")

job_id=""
used_gpu=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gres=gpu:$GPU_REQUEST"
  if output="$(sbatch "${SBATCH_ARGS[@]}" --gres="gpu:$GPU_REQUEST" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_table1_gsk3b_n20_pilot.sh'")"; then
    job_id="${output%%;*}"
    used_gpu="$GPU_REQUEST"
    break
  fi
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_table1_gsk3b_n20_pilot.sh'")"; then
    job_id="${output%%;*}"
    used_gpu="$GPU_REQUEST"
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit GSK3B n=20 pilot" >&2
  exit 1
fi

echo "gsk3b_n20_pilot_job=$job_id"
echo "gpu=$used_gpu"
echo "log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
