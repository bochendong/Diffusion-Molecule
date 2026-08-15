#!/usr/bin/env bash
# Submit B24 on one 10GB H100 MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_FRAGMENT_KERNEL_SEED:-1761}"
LOG_DIR="${SUCC_FRAGMENT_KERNEL_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/latent_fragment_attachment_kernel_v24}"
MAIL_USER="${SUCC_FRAGMENT_KERNEL_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-fragment-kernel-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_FRAGMENT_KERNEL_TIME:-00:15:00}" \
  --cpus-per-task=1 \
  --mem=6G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-fragment-kernel-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_latent_fragment_attachment_kernel_pilot.sh'")"

echo "$submission"
echo "latent_fragment_attachment_kernel_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/latent_fragment_attachment_kernel_v24/seed_${SEED}/summary.json"
