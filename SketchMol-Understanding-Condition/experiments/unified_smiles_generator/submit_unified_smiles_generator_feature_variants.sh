#!/usr/bin/env bash
# Submit full/text-only frozen condition-feature export to Slurm.

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
ACCOUNT="${SUCC_UNIFIED_FEATURE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_FEATURE_SLURM_TIME:-${SUCC_SLURM_TIME:-12:00:00}}"
MEM="${SUCC_UNIFIED_FEATURE_SLURM_MEM:-${SUCC_SLURM_MEM:-96G}}"
CPUS="${SUCC_UNIFIED_FEATURE_SLURM_CPUS:-${SUCC_SLURM_CPUS:-8}}"
JOB_NAME="${SUCC_UNIFIED_FEATURE_SLURM_JOB_NAME:-succ-unified-smiles-features}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_UNIFIED_FEATURE_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
GPU_PROFILE="${SUCC_UNIFIED_FEATURE_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_40gb_mig}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_UNIFIED_FEATURE_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_UNIFIED_FEATURE_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting unified SMILES condition-feature variants"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  train_csv=${SUCC_UNIFIED_TRAIN_CSV:-none}"
echo "  eval_csv=${SUCC_UNIFIED_EVAL_CSV:-none}"
echo "  output_root=${SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_feature_variants_v1}"
echo "  variants=${SUCC_UNIFIED_FEATURE_VARIANTS:-full,text_only}"
echo "  encoder=${SUCC_UNIFIED_FEATURE_ENCODER:-hf_vlm}"
echo "  hf_model=${SUCC_HF_MODEL_NAME_OR_PATH:-none}"
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
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_unified_smiles_generator_feature_variants.sh'")"; then
    continue
  fi
  echo "$output"
  job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -n "$job_id" ]]; then
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit unified condition-feature variants." >&2
  exit 1
fi

echo "unified_smiles_generator_feature_variants_job=$job_id"
