#!/bin/bash
#SBATCH --account=rrg-hup
# 10GB H100 MIG（Nibi）；整卡用 #SBATCH --gpus=h100:1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./slurm-%j.log
#SBATCH --error=./slurm-%j.log
#SBATCH --mail-user=dongbochen1218@gmail.com
#SBATCH --mail-type=BEGIN

set -euo pipefail
unset LD_LIBRARY_PATH
unset PYTHONPATH

# Align system CUDA libs with PyTorch cu124 wheel; override with MODULE_CUDA=... if needed.
export MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_module_venv.sh"

echo "jobid=${SLURM_JOB_ID:-} node=$(hostname) cwd=$(pwd)"
nvidia-smi || true
python -c "import torch; print('cuda=', torch.cuda.is_available())" || true

# Defaults match ChEMBL download → data/molecules.csv (single "smiles" column).
# Override when submitting, e.g.:
#   PHYSTABMOL_DATA=/path/to.csv PHYSTABMOL_RUN_NAME=my_run sbatch scripts/run_phystabmol_gpu.slurm.sh
PHYSTABMOL_DATA="${PHYSTABMOL_DATA:-data/molecules.csv}"
PHYSTABMOL_BACKEND="${PHYSTABMOL_BACKEND:-torch}"
PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_NAME:-slurm_${SLURM_JOB_ID:-manual}}"
PHYSTABMOL_SAMPLES="${PHYSTABMOL_SAMPLES:-32}"
PHYSTABMOL_DECODE_TOP_K="${PHYSTABMOL_DECODE_TOP_K:-5}"

python -m phystabmol.experiment \
  --data "$PHYSTABMOL_DATA" \
  --backend "$PHYSTABMOL_BACKEND" \
  --run-name "$PHYSTABMOL_RUN_NAME" \
  --samples-per-condition "$PHYSTABMOL_SAMPLES" \
  --decode-top-k "$PHYSTABMOL_DECODE_TOP_K"

echo "PhysTabMol job finished at $(date -Is)"
