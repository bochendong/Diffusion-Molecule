#!/usr/bin/env bash
# Submit B25 on CPU; no training is performed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_TWO_STEP_FRAGMENT_SEED:-1761}"
LOG_DIR="${SUCC_TWO_STEP_FRAGMENT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/two_step_residual_fragment_rollout_v25}"
MAIL_USER="${SUCC_TWO_STEP_FRAGMENT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-two-fragment-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_TWO_STEP_FRAGMENT_TIME:-00:08:00}" \
  --cpus-per-task=1 \
  --mem=4G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-two-fragment-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_two_step_residual_fragment_rollout.sh'")"

echo "$submission"
echo "two_step_residual_fragment_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/two_step_residual_fragment_rollout_v25/seed_${SEED}/summary.json"
