#!/usr/bin/env bash
# Submit the single-seed B31 CPU pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_ASSAY_JOINT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/assay_joint_site_token_latent_v31}"
MAIL_USER="${SUCC_ASSAY_JOINT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-assay-joint-v31" \
  --account=def-hup-ab_cpu \
  --time="01:00:00" \
  --cpus-per-task=8 \
  --mem=32G \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-assay-joint-v31-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_assay_joint_site_token_latent.sh'")"

echo "$submission"
echo "assay_joint_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/assay_joint_site_token_latent_v31/seed_1931/summary.json"
