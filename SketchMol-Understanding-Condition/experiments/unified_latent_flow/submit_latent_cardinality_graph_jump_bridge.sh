#!/usr/bin/env bash
# Submit one bounded B39 warm-start run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_CARDINALITY_JUMP_SEED:-1987}"
LOG_DIR="${SUCC_CARDINALITY_JUMP_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/latent_cardinality_graph_jump_bridge_v39}"
MAIL_USER="${SUCC_CARDINALITY_JUMP_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-card-jump-v39" \
  --account=def-hup-ab \
  --time="${SUCC_CARDINALITY_JUMP_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_CARDINALITY_JUMP_CPUS:-4}" \
  --mem="${SUCC_CARDINALITY_JUMP_MEM:-20G}" \
  --gres="${SUCC_CARDINALITY_JUMP_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-card-jump-v39-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_latent_cardinality_graph_jump_bridge.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "latent_cardinality_jump_job=$job_id"
echo "log=$LOG_DIR/uca-card-jump-v39-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/latent_cardinality_graph_jump_bridge_v39/seed_${SEED}/summary.json"
