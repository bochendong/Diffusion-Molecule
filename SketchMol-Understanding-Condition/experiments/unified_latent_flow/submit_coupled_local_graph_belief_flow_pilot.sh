#!/usr/bin/env bash
# Submit one bounded 10GB MIG coupled-local graph-belief pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_COUPLED_BELIEF_SEED:-1733}"
LOG_DIR="${SUCC_COUPLED_BELIEF_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/coupled_local_graph_belief_flow_v4}"
MAIL_USER="${SUCC_COUPLED_BELIEF_MAIL_USER:-dongbochen1218@gmail.com}"
OUTPUT_DIR="${SUCC_COUPLED_BELIEF_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/coupled_local_graph_belief_flow_v4/seed_${SEED}}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-coupled-v4-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_COUPLED_BELIEF_TIME:-00:20:00}" \
  --cpus-per-task="${SUCC_COUPLED_BELIEF_CPUS:-4}" \
  --mem="${SUCC_COUPLED_BELIEF_MEM:-16G}" \
  --gres="${SUCC_COUPLED_BELIEF_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-coupled-v4-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_coupled_local_graph_belief_flow_pilot.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "coupled_local_graph_belief_job=$job_id"
echo "log=$LOG_DIR/uca-coupled-v4-s${SEED}-${job_id}.log"
echo "summary=$OUTPUT_DIR/summary.json"
