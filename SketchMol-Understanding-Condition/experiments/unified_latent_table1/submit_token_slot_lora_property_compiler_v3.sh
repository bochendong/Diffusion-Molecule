#!/usr/bin/env bash
# Submit one 1.5B token-slot LoRA signal experiment on a 20GB H100 MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_TOKEN_SLOT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/token_slot_lora_property_compiler_v3}"
MAIL_USER="${SUCC_TOKEN_SLOT_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_TOKEN_SLOT_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_TOKEN_SLOT_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

job_id="$(sbatch --parsable \
  --job-name=uca-token-slot-v3 \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-token-slot-v3-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_token_slot_lora_property_compiler_v3.sh'")"

echo "job_id=$job_id"
echo "gpu=$GPU_REQUEST"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/token_slot_lora_property_compiler_v3/seed_2055/summary.json"
