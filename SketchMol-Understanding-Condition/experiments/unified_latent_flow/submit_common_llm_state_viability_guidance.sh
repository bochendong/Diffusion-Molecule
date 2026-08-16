#!/usr/bin/env bash
# Submit prepare -> parallel critic arms -> CPU merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_STATE_GUIDE_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/common_llm_state_viability_guidance_v1}"
MAIL_USER="${SUCC_STATE_GUIDE_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_STATE_GUIDE_ACCOUNT:-def-hup-ab}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-state-guide-prep \
  --account="$ACCOUNT" \
  --time=01:00:00 \
  --cpus-per-task=4 \
  --mem=20G \
  --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/uca-state-guide-prep-%j.log" \
  --wrap="SUCC_STATE_GUIDE_STAGE=prepare bash '$SCRIPT_DIR/run_common_llm_state_viability_guidance.sh'")"

property_job="$(sbatch --parsable \
  --job-name=uca-state-guide-prop \
  --account="$ACCOUNT" \
  --time=01:00:00 \
  --cpus-per-task=4 \
  --mem=20G \
  --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --dependency="afterok:$prepare_job" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/uca-state-guide-prop-%j.log" \
  --wrap="SUCC_STATE_GUIDE_STAGE=property_memory bash '$SCRIPT_DIR/run_common_llm_state_viability_guidance.sh'")"

llm_job="$(sbatch --parsable \
  --job-name=uca-state-guide-llm \
  --account="$ACCOUNT" \
  --time=01:30:00 \
  --cpus-per-task=4 \
  --mem=32G \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  --dependency="afterok:$prepare_job" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/uca-state-guide-llm-%j.log" \
  --wrap="SUCC_STATE_GUIDE_STAGE=common_llm_memory bash '$SCRIPT_DIR/run_common_llm_state_viability_guidance.sh'")"

merge_job="$(sbatch --parsable \
  --job-name=uca-state-guide-merge \
  --account="$ACCOUNT" \
  --time=00:10:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --dependency="afterany:$property_job:$llm_job" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-state-guide-merge-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_merge_common_llm_state_viability_guidance.sh'")"

echo "prepare_job=$prepare_job"
echo "property_memory_job=$property_job"
echo "common_llm_memory_job=$llm_job"
echo "merge_job=$merge_job"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/common_llm_state_viability_guidance_v1/seed_1995/merged/summary.json"
