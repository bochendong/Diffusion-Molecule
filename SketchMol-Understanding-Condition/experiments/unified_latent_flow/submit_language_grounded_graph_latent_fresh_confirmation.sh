#!/usr/bin/env bash
# Submit prepare -> parallel matched arms -> post-freeze evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_LANG_FRESH_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/language_grounded_graph_latent_fresh_edit_v3}"
MAIL_USER="${SUCC_LANG_FRESH_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_LANG_FRESH_ACCOUNT:-def-hup-ab}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-lang-edit3-prepare \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=4 --mem=24G \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-lang-edit3-prepare-%j.log" \
  --wrap="SUCC_LANG_FRESH_STAGE=prepare bash '$SCRIPT_DIR/run_language_grounded_graph_latent_fresh_confirmation.sh'")"

property_job="$(sbatch --parsable \
  --job-name=uca-lang-edit3-property \
  --account="$ACCOUNT" --time=01:00:00 --cpus-per-task=4 --mem=20G \
  --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-lang-edit3-property-%j.log" \
  --wrap="SUCC_LANG_FRESH_STAGE=arm SUCC_LANG_FRESH_ARM=property_memory bash '$SCRIPT_DIR/run_language_grounded_graph_latent_fresh_confirmation.sh'")"

llm_job="$(sbatch --parsable \
  --job-name=uca-lang-edit3-llm \
  --account="$ACCOUNT" --time=01:15:00 --cpus-per-task=4 --mem=32G \
  --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-lang-edit3-llm-%j.log" \
  --wrap="SUCC_LANG_FRESH_STAGE=arm SUCC_LANG_FRESH_ARM=common_llm_memory bash '$SCRIPT_DIR/run_language_grounded_graph_latent_fresh_confirmation.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-lang-edit3-eval \
  --account="$ACCOUNT" --time=00:20:00 --cpus-per-task=4 --mem=16G \
  --dependency="afterok:$property_job:$llm_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-lang-edit3-eval-%j.log" \
  --wrap="SUCC_LANG_FRESH_STAGE=evaluate bash '$SCRIPT_DIR/run_language_grounded_graph_latent_fresh_confirmation.sh'")"

echo "prepare_job=$prepare_job"
echo "property_memory_job=$property_job"
echo "common_llm_memory_job=$llm_job"
echo "evaluation_job=$evaluate_job"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/language_grounded_graph_latent_fresh_edit_v3/seed_2022/evaluation/summary.json"
