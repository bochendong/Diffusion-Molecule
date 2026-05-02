#!/bin/bash
#SBATCH --job-name=phystabmol-instr
#SBATCH --account=def-hup-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./instruction-%j.log
#SBATCH --error=./instruction-%j.log

set -euo pipefail
unset LD_LIBRARY_PATH
unset PYTHONPATH

# Submit from PhysTabMol repo root:
#   cd .../PhysTabMol
#   sbatch scripts/run_instruction_editing_gpu.slurm.sh

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PHYSTABMOL_ROOT="$SLURM_SUBMIT_DIR"
else
  PHYSTABMOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$PHYSTABMOL_ROOT/scripts"
cd "$PHYSTABMOL_ROOT"

export MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"
if command -v module >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/env_module_venv.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env_module_venv.sh"
fi

echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
nvidia-smi || true
python3 -c "import torch; print('cuda=', torch.cuda.is_available())" || true

DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
if [[ ! -s "$DATASET" ]]; then
  echo "Instruction dataset not found at $DATASET; building it first."
  bash scripts/build_instruction_dataset.sh
fi

python3 -m phystabmol.instruction_experiment \
  --dataset "$DATASET" \
  --backend "${PHYSTABMOL_BACKEND:-torch}" \
  --run-name "${PHYSTABMOL_RUN_NAME:-instruction_slurm_${SLURM_JOB_ID:-manual}}" \
  --limit "${PHYSTABMOL_LIMIT:-0}" \
  --eval-limit "${PHYSTABMOL_EVAL_LIMIT:-2000}" \
  --samples-per-instruction "${PHYSTABMOL_SAMPLES:-8}" \
  --decode-top-k "${PHYSTABMOL_DECODE_TOP_K:-2}" \
  --torch-epochs "${PHYSTABMOL_TORCH_EPOCHS:-80}" \
  --torch-batch-size "${PHYSTABMOL_TORCH_BATCH_SIZE:-1024}" \
  --torch-hidden-dim "${PHYSTABMOL_TORCH_HIDDEN_DIM:-1024}" \
  --torch-layers "${PHYSTABMOL_TORCH_LAYERS:-6}" \
  --timesteps "${PHYSTABMOL_TIMESTEPS:-80}" \
  --noise-repeats "${PHYSTABMOL_NOISE_REPEATS:-8}"

echo "Instruction editing job finished at $(date -Is)"
