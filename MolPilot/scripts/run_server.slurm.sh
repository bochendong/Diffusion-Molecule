#!/bin/bash
#SBATCH --job-name=molpilot
#SBATCH --account=def-hup-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./molpilot-%j.log
#SBATCH --error=./molpilot-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
nvidia-smi || true

bash scripts/run_server.sh

