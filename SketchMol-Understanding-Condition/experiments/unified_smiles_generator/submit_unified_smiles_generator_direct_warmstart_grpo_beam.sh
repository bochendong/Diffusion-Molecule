#!/usr/bin/env bash
# Submit the direct-checkpoint warm-start + GRPO + beam rescue run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_TIME:-24:00:00}"
MEM="${SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_MEM:-80G}"
CPUS="${SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_CPUS:-8}"
GPU_PROFILE="${SUCC_UNIFIED_DIRECT_WARMSTART_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_40gb_mig}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
JOB_NAME="${SUCC_UNIFIED_DIRECT_WARMSTART_JOB_NAME:-succ-unified-direct-grpo-beam}"
PARTITION="${SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_UNIFIED_DIRECT_WARMSTART_SLURM_GPUS")
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

echo "Submitting unified direct warm-start GRPO beam"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  gpu_candidates=${GPU_CANDIDATES[*]:-none}"
echo "  output_root=${SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_direct_warmstart_grpo_beam_v1}"
echo "  direct_checkpoint=${SUCC_UNIFIED_DIRECT_WARMSTART_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
echo "  condition_layout=${SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT:-direct_compat}"
echo "  objective=${SUCC_UNIFIED_RL_OBJECTIVE:-grpo}"
echo "  benchmark_tasks=${SUCC_UNIFIED_DIRECT_WARMSTART_BENCHMARK_TASKS:-denovo_2p7p,denovo_ood}"
echo "  beam_size=${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE:-40}"

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
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_unified_smiles_generator_direct_warmstart_grpo_beam.sh'")"; then
      continue
    fi
  else
    echo "Trying sbatch without GPU request"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$SCRIPT_DIR/run_unified_smiles_generator_direct_warmstart_grpo_beam.sh'")"; then
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
  echo "ERROR: failed to submit unified direct warm-start GRPO beam." >&2
  exit 1
fi

echo "unified_direct_warmstart_grpo_beam_job=$job_id"
