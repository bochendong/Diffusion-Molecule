#!/usr/bin/env bash
# CPU fresh selection -> paired H100 array -> CPU evaluation -> complete science artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_V9_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/semantic_intervention_fresh_v9}"
MAIL_USER="${SUCC_V9_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_V9_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_V9_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-v9-semantic-prepare \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v9-semantic-prepare-%j.log" \
  --wrap="SUCC_V9_STAGE=prepare bash '$SCRIPT_DIR/run_semantic_intervention_fresh_v9.sh'")"

freeze_job="$(sbatch --parsable \
  --job-name=uca-v9-semantic-freeze \
  --account="$ACCOUNT" --time=01:30:00 --cpus-per-task=4 --mem=28G \
  --gres="gpu:$GPU_REQUEST" --array=0-2%2 \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v9-semantic-freeze-%A_%a.log" \
  --wrap="SUCC_V9_STAGE=freeze bash '$SCRIPT_DIR/run_semantic_intervention_fresh_v9.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-v9-semantic-eval \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=2 --mem=8G \
  --array=0-2%3 --dependency="afterok:$freeze_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v9-semantic-eval-%A_%a.log" \
  --wrap="SUCC_V9_STAGE=evaluate bash '$SCRIPT_DIR/run_semantic_intervention_fresh_v9.sh'")"

gate_job="$(sbatch --parsable \
  --job-name=uca-v9-semantic-scigate \
  --account="$ACCOUNT" --time=00:10:00 --cpus-per-task=1 --mem=4G \
  --dependency="afterok:$evaluate_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-v9-semantic-scigate-%j.log" \
  --wrap="SUCC_V9_STAGE=gate bash '$SCRIPT_DIR/run_semantic_intervention_fresh_v9.sh'")"

echo "prepare_job=$prepare_job"
echo "freeze_job=$freeze_job"
echo "evaluate_job=$evaluate_job"
echo "gate_job=$gate_job"
echo "gpu=$GPU_REQUEST"
echo "candidate_contract=3_replicates_x_6_arms_x_20_fresh_3p_conditions_x_exact_20"
echo "primary_metrics=candidate_property_success_candidate_strict_success_property_fraction"
echo "science_contract=scientific_stop_exits_zero"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/semantic_intervention_fresh_v9/seed_2141/gate/gate_summary.json"
