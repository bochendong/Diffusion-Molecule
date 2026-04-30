#!/bin/bash
#SBATCH --account=rrg-hup
#SBATCH --gpus=h100_2g.20gb:1
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

python -m phystabmol.experiment \
  --backend auto \
  --run-name "slurm_${SLURM_JOB_ID:-manual}" \
  --samples-per-condition 32 \
  --decode-top-k 5

echo "PhysTabMol job finished at $(date -Is)"
