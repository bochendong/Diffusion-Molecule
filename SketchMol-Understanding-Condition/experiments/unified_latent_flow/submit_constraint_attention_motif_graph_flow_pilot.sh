#!/usr/bin/env bash
# Submit the isolated B12 matched cross-attention pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_CONSTRAINT_ATTN_SEED:-1741}"
LOG_DIR="${SUCC_CONSTRAINT_ATTN_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/constraint_attention_motif_graph_flow_v12}"
MAIL_USER="${SUCC_CONSTRAINT_ATTN_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-attn-motif-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_CONSTRAINT_ATTN_TIME:-00:10:00}" \
  --cpus-per-task=4 \
  --mem=16G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-attn-motif-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_constraint_attention_motif_graph_flow_pilot.sh'")"

echo "$submission"
echo "constraint_attention_pilot_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/constraint_attention_motif_graph_flow_v12/seed_${SEED}/summary.json"
