#!/usr/bin/env bash
# Submit one bounded exact-20 set-closed transport signal on a 10GB H100 MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_SET_TRANSPORT_SEED:-2003}"
LOG_DIR="${SUCC_SET_TRANSPORT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/set_closed_graph_transport_v1}"
MAIL_USER="${SUCC_SET_TRANSPORT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-set-transport-v1" \
  --account=def-hup-ab \
  --time="${SUCC_SET_TRANSPORT_TIME:-01:30:00}" \
  --cpus-per-task="${SUCC_SET_TRANSPORT_CPUS:-4}" \
  --mem="${SUCC_SET_TRANSPORT_MEM:-20G}" \
  --gres="${SUCC_SET_TRANSPORT_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-set-transport-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_set_closed_graph_transport.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "set_closed_transport_job=$job_id"
echo "log=$LOG_DIR/uca-set-transport-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/set_closed_graph_transport_v1/seed_${SEED}/summary.json"
