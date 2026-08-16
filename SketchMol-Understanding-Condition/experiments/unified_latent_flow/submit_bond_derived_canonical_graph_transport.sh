#!/usr/bin/env bash
# Submit one bounded B42 run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_CANONICAL_GRAPH_SEED:-1993}"
LOG_DIR="${SUCC_CANONICAL_GRAPH_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/bond_derived_canonical_graph_transport_v42}"
MAIL_USER="${SUCC_CANONICAL_GRAPH_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-canonical-v42" \
  --account=def-hup-ab \
  --time="${SUCC_CANONICAL_GRAPH_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_CANONICAL_GRAPH_CPUS:-4}" \
  --mem="${SUCC_CANONICAL_GRAPH_MEM:-20G}" \
  --gres="${SUCC_CANONICAL_GRAPH_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-canonical-v42-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_bond_derived_canonical_graph_transport.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "canonical_graph_transport_job=$job_id"
echo "log=$LOG_DIR/uca-canonical-v42-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/bond_derived_canonical_graph_transport_v42/seed_${SEED}/summary.json"
