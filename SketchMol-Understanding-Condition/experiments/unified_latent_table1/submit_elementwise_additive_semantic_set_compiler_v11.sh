#!/usr/bin/env bash
# Submit final V11 execution and separate scientific gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_V11_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/elementwise_additive_semantic_set_compiler_v11}"
MAIL_USER="${SUCC_V11_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_V11_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_V11_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

execution_job_id="$(sbatch --parsable \
  --job-name=uca-elemset-v11-exec \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-elemset-v11-exec-%j.log" \
  --wrap="SUCC_V11_STAGE=execute bash '$SCRIPT_DIR/run_elementwise_additive_semantic_set_compiler_v11.sh'")"

gate_job_id="$(sbatch --parsable \
  --job-name=uca-elemset-v11-scigate \
  --account="$ACCOUNT" --time=00:05:00 --cpus-per-task=1 --mem=2G \
  --dependency="afterok:$execution_job_id" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-elemset-v11-scigate-%j.log" \
  --wrap="SUCC_V11_STAGE=gate bash '$SCRIPT_DIR/run_elementwise_additive_semantic_set_compiler_v11.sh'")"

echo "execution_job_id=$execution_job_id"
echo "gate_job_id=$gate_job_id"
echo "final_attempt=true"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/elementwise_additive_semantic_set_compiler_v11/seed_2161/gate/gate_summary.json"
