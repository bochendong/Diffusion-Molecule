#!/usr/bin/env bash
# Submit the B24 evidence gate on CPU only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_FRAGMENT_COVERAGE_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/fragment_attachment_coverage_v24}"
MAIL_USER="${SUCC_FRAGMENT_COVERAGE_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-fragment-coverage" \
  --account=def-hup-ab \
  --time="${SUCC_FRAGMENT_COVERAGE_TIME:-00:12:00}" \
  --cpus-per-task=1 \
  --mem=4G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-fragment-coverage-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_fragment_attachment_coverage_gate.sh'")"

echo "$submission"
echo "fragment_attachment_coverage_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "coverage_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/fragment_attachment_coverage_v24/seed_1741/summary.json"
