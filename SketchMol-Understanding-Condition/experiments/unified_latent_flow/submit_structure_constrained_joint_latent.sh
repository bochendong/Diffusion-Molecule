#!/usr/bin/env bash
# Submit the single-seed B32 CPU pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_STRUCTURE_JOINT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/structure_constrained_joint_latent_v32}"
MAIL_USER="${SUCC_STRUCTURE_JOINT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-structure-joint-v32" \
  --account=def-hup-ab_cpu \
  --time="01:00:00" \
  --cpus-per-task=8 \
  --mem=32G \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-structure-joint-v32-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_structure_constrained_joint_latent.sh'")"

echo "$submission"
echo "structure_joint_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/structure_constrained_joint_latent_v32/seed_1941/summary.json"
