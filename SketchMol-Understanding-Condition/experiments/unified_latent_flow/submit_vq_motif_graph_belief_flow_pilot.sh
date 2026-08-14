#!/usr/bin/env bash
# Submit one bounded 10GB MIG VQ-motif graph-belief pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_VQ_MOTIF_SEED:-1737}"
LOG_DIR="${SUCC_VQ_MOTIF_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/vq_motif_graph_belief_flow_v5}"
OUTPUT_DIR="${SUCC_VQ_MOTIF_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/vq_motif_graph_belief_flow_v5/seed_${SEED}}"
MAIL_USER="${SUCC_VQ_MOTIF_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-vq-motif-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_VQ_MOTIF_TIME:-00:20:00}" \
  --cpus-per-task="${SUCC_VQ_MOTIF_CPUS:-4}" \
  --mem="${SUCC_VQ_MOTIF_MEM:-16G}" \
  --gres="${SUCC_VQ_MOTIF_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-vq-motif-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_vq_motif_graph_belief_flow_pilot.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "vq_motif_graph_belief_job=$job_id"
echo "log=$LOG_DIR/uca-vq-motif-s${SEED}-${job_id}.log"
echo "summary=$OUTPUT_DIR/summary.json"
