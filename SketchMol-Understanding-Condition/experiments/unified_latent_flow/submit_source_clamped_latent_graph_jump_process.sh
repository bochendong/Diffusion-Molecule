#!/usr/bin/env bash
# Submit one bounded B38 run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_GRAPH_JUMP_SEED:-1985}"
LOG_DIR="${SUCC_GRAPH_JUMP_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/source_clamped_latent_graph_jump_process_v38}"
MAIL_USER="${SUCC_GRAPH_JUMP_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-graph-jump-v38" \
  --account=def-hup-ab \
  --time="${SUCC_GRAPH_JUMP_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_GRAPH_JUMP_CPUS:-4}" \
  --mem="${SUCC_GRAPH_JUMP_MEM:-20G}" \
  --gres="${SUCC_GRAPH_JUMP_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-graph-jump-v38-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_source_clamped_latent_graph_jump_process.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "source_clamped_graph_jump_job=$job_id"
echo "log=$LOG_DIR/uca-graph-jump-v38-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/source_clamped_latent_graph_jump_process_v38/seed_${SEED}/summary.json"
