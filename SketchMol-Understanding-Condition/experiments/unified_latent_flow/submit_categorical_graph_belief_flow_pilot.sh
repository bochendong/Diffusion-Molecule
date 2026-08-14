#!/usr/bin/env bash
# Submit one bounded H100 10GB MIG native categorical graph-belief pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_GRAPH_BELIEF_SEED:-1731}"
LOG_DIR="${SUCC_GRAPH_BELIEF_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/categorical_graph_belief_flow_v3}"
MAIL_USER="${SUCC_GRAPH_BELIEF_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-belief-v3-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_GRAPH_BELIEF_TIME:-00:20:00}" \
  --cpus-per-task="${SUCC_GRAPH_BELIEF_CPUS:-4}" \
  --mem="${SUCC_GRAPH_BELIEF_MEM:-16G}" \
  --gres="${SUCC_GRAPH_BELIEF_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-belief-v3-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_categorical_graph_belief_flow_pilot.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "categorical_graph_belief_job=$job_id"
echo "log=$LOG_DIR/uca-belief-v3-s${SEED}-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/categorical_graph_belief_flow_v3/seed_${SEED}/summary.json"
