#!/usr/bin/env bash
# Submit one bounded 10GB MIG connected-region hierarchical VQ ablation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_REGION_HIER_VQ_SEED:-1741}"
LOG_DIR="${SUCC_REGION_HIER_VQ_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/connected_region_hierarchical_vq_graph_flow_v8}"
OUTPUT_DIR="${SUCC_REGION_HIER_VQ_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/connected_region_hierarchical_vq_graph_flow_v8/seed_${SEED}}"
MAIL_USER="${SUCC_REGION_HIER_VQ_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-region-hvq-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_REGION_HIER_VQ_TIME:-00:20:00}" \
  --cpus-per-task="${SUCC_REGION_HIER_VQ_CPUS:-4}" \
  --mem="${SUCC_REGION_HIER_VQ_MEM:-16G}" \
  --gres="${SUCC_REGION_HIER_VQ_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-region-hvq-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_connected_region_hierarchical_vq_graph_flow_pilot.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "connected_region_hierarchical_vq_job=$job_id"
echo "log=$LOG_DIR/uca-region-hvq-s${SEED}-${job_id}.log"
echo "summary=$OUTPUT_DIR/summary.json"
