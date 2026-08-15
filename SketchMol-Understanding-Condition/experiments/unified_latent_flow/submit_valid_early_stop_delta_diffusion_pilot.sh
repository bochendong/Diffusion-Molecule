#!/usr/bin/env bash
# Submit B22 valid early-stop delta diffusion on one 10GB H100 MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_EARLY_STOP_DIFFUSION_SEED:-1757}"
LOG_DIR="${SUCC_EARLY_STOP_DIFFUSION_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/valid_early_stop_delta_diffusion_v22}"
MAIL_USER="${SUCC_EARLY_STOP_DIFFUSION_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-early-stop-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_EARLY_STOP_DIFFUSION_TIME:-00:25:00}" \
  --cpus-per-task=1 \
  --mem=8G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-early-stop-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_valid_early_stop_delta_diffusion_pilot.sh'")"

echo "$submission"
echo "valid_early_stop_delta_diffusion_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/valid_early_stop_delta_diffusion_v22/seed_${SEED}/summary.json"
