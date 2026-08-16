#!/usr/bin/env bash
# Submit four parallel latent-controller arms and one dependent CPU merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_LLM_LATENT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/common_llm_latent_operator_signal_v1}"
MAIL_USER="${SUCC_LLM_LATENT_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_LLM_LATENT_ACCOUNT:-def-hup-ab}"
mkdir -p "$LOG_DIR"

submit_arm() {
  local arm="$1"
  local time_limit="$2"
  local memory="$3"
  local gres="$4"
  sbatch --parsable \
    --job-name="uca-llm-lat-${arm}" \
    --account="$ACCOUNT" \
    --time="$time_limit" \
    --cpus-per-task=4 \
    --mem="$memory" \
    --gres="gpu:${gres}" \
    --mail-user="$MAIL_USER" \
    --mail-type=FAIL \
    --output="$LOG_DIR/uca-llm-lat-${arm}-%j.log" \
    --wrap="SUCC_LLM_LATENT_ARM='$arm' bash '$SCRIPT_DIR/run_llm_latent_operator_signal.sh'"
}

property_job="$(submit_arm property_mlp 00:45:00 20G nvidia_h100_80gb_hbm3_1g.10gb:1)"
base_job="$(submit_arm base_frozen 01:15:00 32G nvidia_h100_80gb_hbm3_2g.20gb:1)"
sft_job="$(submit_arm sft_frozen 01:15:00 32G nvidia_h100_80gb_hbm3_2g.20gb:1)"
lora_job="$(submit_arm sft_lora 02:00:00 64G nvidia_h100_80gb_hbm3_3g.40gb:1)"
dependency="afterany:${property_job}:${base_job}:${sft_job}:${lora_job}"
merge_job="$(sbatch --parsable \
  --job-name="uca-llm-lat-merge" \
  --account="$ACCOUNT" \
  --time=00:10:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --dependency="$dependency" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-llm-lat-merge-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_merge_llm_latent_operator_signal.sh'")"

echo "property_mlp_job=$property_job"
echo "base_frozen_job=$base_job"
echo "sft_frozen_job=$sft_job"
echo "sft_lora_job=$lora_job"
echo "merge_job=$merge_job"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/common_llm_latent_operator_signal_v1/seed_1993/merged/summary.json"
