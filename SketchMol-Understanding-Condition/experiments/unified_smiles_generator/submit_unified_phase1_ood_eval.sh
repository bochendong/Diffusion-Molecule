#!/usr/bin/env bash
# Submit fixed direct_compat OOD de novo eval (sample x128, full OOD eval rows).

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
ACCOUNT="${SUCC_UNIFIED_PHASE1_OOD_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_PHASE1_OOD_SLURM_TIME:-3:00:00}"
MEM="${SUCC_UNIFIED_PHASE1_OOD_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_PHASE1_OOD_SLURM_CPUS:-8}"
GPU_PROFILE="${SUCC_UNIFIED_PHASE1_OOD_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_full}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_UNIFIED_PHASE1_OOD_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
JOB_NAME="${SUCC_UNIFIED_PHASE1_OOD_JOB_NAME:-succ-unified-phase1-ood-direct-compat}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_UNIFIED_PHASE1_OOD_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_UNIFIED_PHASE1_OOD_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/${JOB_NAME}-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

echo "Submitting unified phase1 OOD direct_compat eval"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  python=$SUCC_PYTHON_BIN"
echo "  output_root=${SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_phase1_ood_direct_compat_v1}"
echo "  num_samples=${SUCC_UNIFIED_NUM_SAMPLES:-128}"

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  output=""
  if [[ -n "$GPU_REQUEST" ]]; then
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_unified_phase1_ood_eval.sh'")"; then
      continue
    fi
  else
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$SCRIPT_DIR/run_unified_phase1_ood_eval.sh'")"; then
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
  echo "ERROR: failed to submit phase1 OOD eval." >&2
  exit 1
fi
echo "unified_phase1_ood_eval_job=$job_id"
