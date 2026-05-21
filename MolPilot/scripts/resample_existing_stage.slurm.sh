#!/bin/bash
#SBATCH --job-name=molpilot-resample
#SBATCH --account=def-hup-ab
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./molpilot-resample-%j.log
#SBATCH --error=./molpilot-resample-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHON_BIN
echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
echo "python_bin=$PYTHON_BIN"
nvidia-smi || true
"$PYTHON_BIN" -c "import sys; print('python=', sys.executable)"
"$PYTHON_BIN" -c "import numpy; print('numpy=', numpy.__version__)"
"$PYTHON_BIN" -c "import torch; print('cuda=', torch.cuda.is_available())" || true

bash scripts/resample_existing_stage.sh
