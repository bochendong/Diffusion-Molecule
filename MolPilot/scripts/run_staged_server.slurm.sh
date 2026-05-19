#!/bin/bash
#SBATCH --job-name=molpilot-staged
#SBATCH --account=def-hup-ab
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./molpilot-staged-%j.log
#SBATCH --error=./molpilot-staged-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
echo "python_bin=$PYTHON_BIN"
nvidia-smi || true
"$PYTHON_BIN" -c "import sys; print('python=', sys.executable)"
"$PYTHON_BIN" -c "import numpy; print('numpy=', numpy.__version__)"
"$PYTHON_BIN" -c "import torch; print('cuda=', torch.cuda.is_available())" || true

bash scripts/run_staged_server.sh
