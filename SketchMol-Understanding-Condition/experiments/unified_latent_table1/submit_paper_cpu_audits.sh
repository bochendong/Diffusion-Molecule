#!/usr/bin/env bash
# Submit two independent CPU-only audits over existing frozen Table1 candidates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_PAPER_AUDIT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/paper_cpu_audits_v1}"
MAIL_USER="${SUCC_PAPER_AUDIT_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_PAPER_AUDIT_ACCOUNT:-def-hup-ab}"
mkdir -p "$LOG_DIR"

particle_job="$(sbatch --parsable \
  --job-name=uca-particle-finalize \
  --account="$ACCOUNT" --time=01:00:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-particle-finalize-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_finalize_existing_particle_coverage.sh'")"

anyk_job="$(sbatch --parsable \
  --job-name=uca-table1-anyk-audit \
  --account="$ACCOUNT" --time=01:00:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-table1-anyk-audit-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_table1_anyk_robustness.sh'")"

echo "particle_finalize_job=$particle_job"
echo "particle_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/b41_particle_coverage_table1_n20/summary.json"
echo "table1_anyk_job=$anyk_job"
echo "table1_anyk_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/table1_anyk_robustness_v1/summary.json"
