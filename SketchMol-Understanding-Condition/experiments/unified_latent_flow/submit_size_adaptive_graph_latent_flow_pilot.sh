#!/usr/bin/env bash
# Submit one bounded H100 10GB MIG size-adaptive graph-flow pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_GRAPH_FLOW_SEED:-1729}"
LOG_DIR="${SUCC_GRAPH_FLOW_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/size_adaptive_graph_latent_flow_v2}"
MAIL_USER="${SUCC_GRAPH_FLOW_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-size-flow-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_GRAPH_FLOW_TIME:-00:20:00}" \
  --cpus-per-task="${SUCC_GRAPH_FLOW_CPUS:-4}" \
  --mem="${SUCC_GRAPH_FLOW_MEM:-16G}" \
  --gres="${SUCC_GRAPH_FLOW_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-size-flow-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_size_adaptive_graph_latent_flow_pilot.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "size_adaptive_graph_flow_job=$job_id"
echo "log=$LOG_DIR/uca-size-flow-s${SEED}-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/size_adaptive_graph_latent_flow_v2/seed_${SEED}/summary.json"
