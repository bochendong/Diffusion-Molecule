#!/bin/bash
#SBATCH --job-name=phystabmol-h100
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
cd "${SLURM_SUBMIT_DIR}"

# Nibi / 集群请按文档加载，例如：
# module load StdEnv cuda cudnn python/3.12

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

exec python3 -m phystabmol.experiment \
  --backend torch \
  --device cuda \
  --run-name "slurm_${SLURM_JOB_ID}" \
  --contrastive-epochs 600 \
  --torch-epochs 200 \
  --torch-batch-size 1024 \
  --torch-hidden-dim 1024 \
  --torch-layers 6 \
  --understanding-backbone clip \
  --understanding-model openai/clip-vit-base-patch32 \
  --timesteps 100 \
  --noise-repeats 24 \
  --samples-per-condition 32 \
  --decode-top-k 5 \
  --run-sketchmol-benchmark \
  --benchmark-samples-per-condition 1000 \
  --benchmark-multi-conditions 1000 \
  --benchmark-optimization-conditions 100 \
  --enable-3d \
  --save-3d-sdf
