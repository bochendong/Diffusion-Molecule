#!/bin/bash
#SBATCH --account=def-hup-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./structure-resume-%j.log
#SBATCH --error=./structure-resume-%j.log

set -euo pipefail

# Slurm copies this script to /var/spool/slurmd/.../slurm_script, so
# dirname(BASH_SOURCE) is only reliable before submission.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$ROOT/scripts"
cd "$ROOT"

if [[ -z "${SLURM_JOB_ID:-}" && "${PHYSTABMOL_SUBMIT:-1}" == "1" ]]; then
  RUN_DIR="${PHYSTABMOL_RUN_DIR:-$(find runs -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)}"
  echo "Submitting structure prompt resume for run_dir=$RUN_DIR"
  PHYSTABMOL_RUN_DIR="$RUN_DIR" sbatch --export=ALL --time="${PHYSTABMOL_WALLTIME:-08:00:00}" "$0" "$@"
  exit 0
fi

unset LD_LIBRARY_PATH
unset PYTHONPATH

export MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_module_venv.sh"

RUN_DIR="${PHYSTABMOL_RUN_DIR:-$(find runs -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)}"
CONDITIONS="${PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS:-0}"
SAMPLES="${PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES:-0}"
DECODE_TOP_K="${PHYSTABMOL_STRUCTURE_PROMPT_DECODE_TOP_K:-0}"

echo "jobid=${SLURM_JOB_ID:-} node=$(hostname) cwd=$(pwd)"
echo "Resuming structure prompt benchmark for $RUN_DIR"
nvidia-smi || true

python -m phystabmol.resume_structure_prompt_benchmark \
  --run-dir "$RUN_DIR" \
  --structure-prompt-conditions "$CONDITIONS" \
  --structure-prompt-samples "$SAMPLES" \
  --structure-prompt-decode-top-k "$DECODE_TOP_K" \
  "$@"

echo "Structure prompt resume finished at $(date -Is)"
