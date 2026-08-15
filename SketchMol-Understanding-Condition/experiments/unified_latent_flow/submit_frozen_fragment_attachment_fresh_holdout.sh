#!/usr/bin/env bash
# Submit the frozen B24 once-only fresh-heldout check on a small CPU allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_FRESH_HOLDOUT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/frozen_fragment_attachment_fresh_holdout_v26}"
MAIL_USER="${SUCC_FRESH_HOLDOUT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-fragment-fresh-b26" \
  --account=def-hup-ab_cpu \
  --time="00:15:00" \
  --cpus-per-task=2 \
  --mem=8G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-fragment-fresh-b26-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_frozen_fragment_attachment_fresh_holdout.sh'")"

echo "$submission"
echo "fresh_holdout_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "fresh_holdout_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/frozen_fragment_attachment_fresh_holdout_v26/seed_1873/summary.json"
