#!/bin/bash
#SBATCH --job-name=phystabmol-h100
#SBATCH --account=rrg-hup
# 10GB H100 MIG；整卡: #SBATCH --gpus=h100:1。CLIP+3D 若 OOM 可改更大 slice 或减 batch
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
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

DATA_PATH="${PHYSTABMOL_DATA:-data/molecules.csv}"
if [[ ! -s "${DATA_PATH}" ]]; then
  echo "Dataset not found at ${DATA_PATH}; downloading ChEMBL first."
  PHYSTABMOL_OUT="${DATA_PATH}" bash scripts/download_chembl_100k.sh --rdkit-filter
fi

exec python3 -m phystabmol.experiment \
  --data "${DATA_PATH}" \
  --backend torch \
  --device cuda \
  --run-name "slurm_${SLURM_JOB_ID}" \
  --contrastive-epochs 600 \
  --contrastive-batch-size 512 \
  --contrastive-max-pairs 20000 \
  --contrastive-retrieval-samples 2048 \
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
