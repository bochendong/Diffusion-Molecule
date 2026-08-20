#!/usr/bin/env bash
# Submit prepare -> parallel graph/language freeze -> sealed evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_FRESH_CONFIRM_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/fresh_graph_jump_language_confirmation_v1}"
MAIL_USER="${SUCC_FRESH_CONFIRM_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_FRESH_CONFIRM_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_FRESH_CONFIRM_GPU:-nvidia_h100_80gb_hbm3_1g.10gb:1}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-fresh-confirm-prepare \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-fresh-confirm-prepare-%j.log" \
  --wrap="SUCC_FRESH_CONFIRM_STAGE=prepare bash '$SCRIPT_DIR/run_fresh_graph_jump_language_confirmation.sh'")"

graph_job="$(sbatch --parsable \
  --job-name=uca-fresh-confirm-graph \
  --account="$ACCOUNT" --time=01:00:00 --cpus-per-task=4 --mem=20G \
  --gres="gpu:$GPU_REQUEST" \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-fresh-confirm-graph-%j.log" \
  --wrap="SUCC_FRESH_CONFIRM_STAGE=freeze SUCC_FRESH_CONFIRM_ARM_GROUP=graph bash '$SCRIPT_DIR/run_fresh_graph_jump_language_confirmation.sh'")"

language_job="$(sbatch --parsable \
  --job-name=uca-fresh-confirm-language \
  --account="$ACCOUNT" --time=01:15:00 --cpus-per-task=4 --mem=20G \
  --gres="gpu:$GPU_REQUEST" \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-fresh-confirm-language-%j.log" \
  --wrap="SUCC_FRESH_CONFIRM_STAGE=freeze SUCC_FRESH_CONFIRM_ARM_GROUP=language bash '$SCRIPT_DIR/run_fresh_graph_jump_language_confirmation.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-fresh-confirm-eval \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=4 --mem=16G \
  --dependency="afterok:$graph_job:$language_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-fresh-confirm-eval-%j.log" \
  --wrap="SUCC_FRESH_CONFIRM_STAGE=evaluate bash '$SCRIPT_DIR/run_fresh_graph_jump_language_confirmation.sh'")"

echo "prepare_job=$prepare_job"
echo "graph_job=$graph_job"
echo "language_job=$language_job"
echo "evaluate_job=$evaluate_job"
echo "gpu=$GPU_REQUEST"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/fresh_graph_jump_language_confirmation_v1/seed_2032/evaluation/summary.json"
