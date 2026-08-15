#!/usr/bin/env bash
# Submit the bounded B23 decoder ablation on one 10GB H100 MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_EDIT_CENTER_SEED:-1759}"
LOG_DIR="${SUCC_EDIT_CENTER_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/latent_edit_center_rewrite_decode_v23}"
MAIL_USER="${SUCC_EDIT_CENTER_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-edit-center-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_EDIT_CENTER_TIME:-00:08:00}" \
  --cpus-per-task=1 \
  --mem=4G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-edit-center-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_latent_edit_center_rewrite_decode.sh'")"

echo "$submission"
echo "latent_edit_center_rewrite_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/latent_edit_center_rewrite_decode_v23/seed_${SEED}/summary.json"
