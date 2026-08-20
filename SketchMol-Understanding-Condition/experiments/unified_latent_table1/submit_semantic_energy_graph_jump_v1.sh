#!/usr/bin/env bash
# CPU prepare -> one 20GB MIG train/freeze allocation -> sealed CPU evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_SEMANTIC_ENERGY_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/semantic_energy_graph_jump_v1}"
MAIL_USER="${SUCC_SEMANTIC_ENERGY_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_SEMANTIC_ENERGY_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_SEMANTIC_ENERGY_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-sem-energy-prepare \
  --account="$ACCOUNT" --time=00:15:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-sem-energy-prepare-%j.log" \
  --wrap="SUCC_SEMANTIC_ENERGY_STAGE=prepare bash '$SCRIPT_DIR/run_semantic_energy_graph_jump_v1.sh'")"

train_freeze_job="$(sbatch --parsable \
  --job-name=uca-sem-energy-trainfreeze \
  --account="$ACCOUNT" --time=02:00:00 --cpus-per-task=4 --mem=32G \
  --gres="gpu:$GPU_REQUEST" \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-sem-energy-trainfreeze-%j.log" \
  --wrap="SUCC_SEMANTIC_ENERGY_STAGE=train bash '$SCRIPT_DIR/run_semantic_energy_graph_jump_v1.sh' && SUCC_SEMANTIC_ENERGY_STAGE=freeze bash '$SCRIPT_DIR/run_semantic_energy_graph_jump_v1.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-sem-energy-eval \
  --account="$ACCOUNT" --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$train_freeze_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-sem-energy-eval-%j.log" \
  --wrap="SUCC_SEMANTIC_ENERGY_STAGE=evaluate bash '$SCRIPT_DIR/run_semantic_energy_graph_jump_v1.sh'")"

echo "prepare_job=$prepare_job"
echo "train_freeze_job=$train_freeze_job"
echo "evaluate_job=$evaluate_job"
echo "gpu=$GPU_REQUEST"
echo "train_gate=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/semantic_energy_graph_jump_v1/seed_2045/train/summary.json"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/semantic_energy_graph_jump_v1/seed_2045/evaluation/summary.json"
