#!/usr/bin/env bash
# Submit one bounded B37 run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_REGION_DIFFUSION_SEED:-1983}"
LOG_DIR="${SUCC_REGION_DIFFUSION_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/source_clamped_region_graph_diffusion_v37}"
MAIL_USER="${SUCC_REGION_DIFFUSION_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-region-diff-v37" \
  --account=def-hup-ab \
  --time="${SUCC_REGION_DIFFUSION_TIME:-01:30:00}" \
  --cpus-per-task="${SUCC_REGION_DIFFUSION_CPUS:-4}" \
  --mem="${SUCC_REGION_DIFFUSION_MEM:-20G}" \
  --gres="${SUCC_REGION_DIFFUSION_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-region-diff-v37-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_source_clamped_region_graph_diffusion.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "source_clamped_region_diffusion_job=$job_id"
echo "log=$LOG_DIR/uca-region-diff-v37-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/source_clamped_region_graph_diffusion_v37/seed_${SEED}/summary.json"
