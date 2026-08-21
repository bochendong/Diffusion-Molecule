#!/usr/bin/env bash
# CPU prepare -> 20GB MIG freeze -> CPU evaluation -> CPU science decision.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_V7_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/mass_conserving_router_fresh_horizon_v7}"
MAIL_USER="${SUCC_V7_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_V7_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_V7_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-v7-fresh-prepare \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=4 --mem=12G \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v7-fresh-prepare-%j.log" \
  --wrap="SUCC_V7_STAGE=prepare bash '$SCRIPT_DIR/run_mass_conserving_router_fresh_horizon_v7.sh'")"

freeze_job="$(sbatch --parsable \
  --job-name=uca-v7-fresh-freeze \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v7-fresh-freeze-%j.log" \
  --wrap="SUCC_V7_STAGE=freeze bash '$SCRIPT_DIR/run_mass_conserving_router_fresh_horizon_v7.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-v7-fresh-eval \
  --account="$ACCOUNT" --time=00:15:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$freeze_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v7-fresh-eval-%j.log" \
  --wrap="SUCC_V7_STAGE=evaluate bash '$SCRIPT_DIR/run_mass_conserving_router_fresh_horizon_v7.sh'")"

gate_job="$(sbatch --parsable \
  --job-name=uca-v7-fresh-scigate \
  --account="$ACCOUNT" --time=00:05:00 --cpus-per-task=1 --mem=2G \
  --dependency="afterok:$evaluate_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-v7-fresh-scigate-%j.log" \
  --wrap="SUCC_V7_STAGE=gate bash '$SCRIPT_DIR/run_mass_conserving_router_fresh_horizon_v7.sh'")"

echo "prepare_job=$prepare_job"
echo "freeze_job=$freeze_job"
echo "evaluate_job=$evaluate_job"
echo "gate_job=$gate_job"
echo "gpu=$GPU_REQUEST"
echo "candidate_contract=40_fresh_conditions_x_exact_20_attempts_x_3_arms"
echo "closure_contract=last_materializable_state_terminal_transition_without_retry_or_ranking"
echo "science_contract=scientific_stop_is_a_complete_gate_artifact_not_a_failed_job"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/mass_conserving_router_fresh_horizon_v7/seed_2093/gate/gate_summary.json"
