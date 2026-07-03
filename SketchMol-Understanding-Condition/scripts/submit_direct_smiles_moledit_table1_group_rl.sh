#!/usr/bin/env bash
# Submit source-conditioned direct-SMILES group RL on MolEdit Table1 tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_TIME:-${SUCC_SLURM_TIME:-24:00:00}}"
MEM="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_MEM:-${SUCC_SLURM_MEM:-96G}}"
CPUS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_CPUS:-${SUCC_SLURM_CPUS:-8}}"
JOB_NAME="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_JOB_NAME:-succ-moledit-table1-group-rl}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
GPU_PROFILE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_40gb_mig}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_DIRECT_MOLEDIT_GROUP_RL_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "none" || "$GPU_PROFILE" == "0" ]]; then
  GPU_CANDIDATES=("")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting direct-SMILES MolEdit Table1 group RL"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  output_dir=${SUCC_DIRECT_MOLEDIT_GROUP_RL_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_moledit_table1_group_rl_v1}"
echo "  train_per_task=${SUCC_DIRECT_MOLEDIT_GROUP_RL_TRAIN_PER_TASK:-500}"
echo "  eval_per_task=${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_PER_TASK:-100}"
echo "  run_sft=${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_SFT:-1}"
echo "  run_rl=${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_RL:-1}"
echo "  rl_rollouts=${SUCC_DIRECT_MOLEDIT_GROUP_RL_ROLLOUTS_PER_PROMPT:-16}"
echo "  rl_advantage_mode=${SUCC_DIRECT_MOLEDIT_GROUP_RL_ADVANTAGE_MODE:-group_zscore}"
echo "  benchmark_num_samples=${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_NUM_SAMPLES:-256}"
echo "  benchmark_budgets=${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_BUDGETS:-20 256}"
echo "  gpu_candidates=${GPU_CANDIDATES[*]:-none}"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  if [[ -n "$GPU_REQUEST" ]]; then
    echo "Trying sbatch with --gpus=$GPU_REQUEST"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_moledit_table1_group_rl.sh'")"; then
      continue
    fi
  else
    echo "Trying sbatch without GPU request"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_direct_smiles_moledit_table1_group_rl.sh'")"; then
      continue
    fi
  fi
  echo "$output"
  job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -n "$job_id" ]]; then
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit direct-SMILES MolEdit Table1 group RL." >&2
  exit 1
fi

echo "direct_smiles_moledit_table1_group_rl_job=$job_id"
