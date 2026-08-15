#!/usr/bin/env bash
# Submit one CPU-only post-freeze B37 evaluation recovery.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/source_clamped_region_graph_diffusion_v37"
MAIL_USER="${SUCC_REGION_RESUME_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-region-v37-eval" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_REGION_RESUME_TIME:-00:15:00}" \
  --cpus-per-task="${SUCC_REGION_RESUME_CPUS:-4}" \
  --mem="${SUCC_REGION_RESUME_MEM:-12G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-region-v37-eval-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_source_clamped_region_graph_diffusion_evaluation_resume.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "source_clamped_region_evaluation_resume_job=$job_id"
echo "log=$LOG_DIR/uca-region-v37-eval-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/source_clamped_region_graph_diffusion_v37/seed_1983/summary.json"
