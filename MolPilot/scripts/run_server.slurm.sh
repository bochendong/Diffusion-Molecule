#!/bin/bash
#SBATCH --job-name=molpilot
#SBATCH --account=def-hup-ab
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --mem-per-cpu=4096M
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./molpilot-%j.log
#SBATCH --error=./molpilot-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
echo "python_bin=$PYTHON_BIN"
nvidia-smi || true

bash scripts/run_server.sh
