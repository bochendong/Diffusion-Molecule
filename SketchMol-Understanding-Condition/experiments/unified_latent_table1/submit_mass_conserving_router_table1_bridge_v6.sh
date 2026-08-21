#!/usr/bin/env bash
# CPU prepare -> 20GB MIG freeze -> CPU evaluation -> CPU science decision.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_V6_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/mass_conserving_router_table1_bridge_v6}"
MAIL_USER="${SUCC_V6_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_V6_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_V6_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

prepare_job="$(sbatch --parsable \
  --job-name=uca-v6-bridge-prepare \
  --account="$ACCOUNT" --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v6-bridge-prepare-%j.log" \
  --wrap="SUCC_V6_STAGE=prepare bash '$SCRIPT_DIR/run_mass_conserving_router_table1_bridge_v6.sh'")"

freeze_job="$(sbatch --parsable \
  --job-name=uca-v6-bridge-freeze \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v6-bridge-freeze-%j.log" \
  --wrap="SUCC_V6_STAGE=freeze bash '$SCRIPT_DIR/run_mass_conserving_router_table1_bridge_v6.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-v6-bridge-eval \
  --account="$ACCOUNT" --time=00:15:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$freeze_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v6-bridge-eval-%j.log" \
  --wrap="SUCC_V6_STAGE=evaluate bash '$SCRIPT_DIR/run_mass_conserving_router_table1_bridge_v6.sh'")"

gate_job="$(sbatch --parsable \
  --job-name=uca-v6-bridge-scigate \
  --account="$ACCOUNT" --time=00:05:00 --cpus-per-task=1 --mem=2G \
  --dependency="afterok:$evaluate_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-v6-bridge-scigate-%j.log" \
  --wrap="SUCC_V6_STAGE=gate bash '$SCRIPT_DIR/run_mass_conserving_router_table1_bridge_v6.sh'")"

echo "prepare_job=$prepare_job"
echo "freeze_job=$freeze_job"
echo "evaluate_job=$evaluate_job"
echo "gate_job=$gate_job"
echo "gpu=$GPU_REQUEST"
echo "execution_contract=engineering_success_always_exits_zero"
echo "science_contract=scientific_stop_is_a_complete_gate_artifact_not_a_failed_job"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/mass_conserving_router_table1_bridge_v6/seed_2081/gate/gate_summary.json"
