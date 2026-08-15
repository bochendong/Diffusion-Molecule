#!/usr/bin/env bash
# Submit the single-seed B33 CPU confirmation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_PARETO_JOINT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/pareto_conditioned_joint_latent_v33}"
MAIL_USER="${SUCC_PARETO_JOINT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-pareto-joint-v33" \
  --account=def-hup-ab_cpu \
  --time="00:30:00" \
  --cpus-per-task=4 \
  --mem=16G \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-pareto-joint-v33-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_pareto_conditioned_joint_latent.sh'")"

echo "$submission"
echo "pareto_joint_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/pareto_conditioned_joint_latent_v33/seed_1951/summary.json"
