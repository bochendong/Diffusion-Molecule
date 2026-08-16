#!/usr/bin/env bash
# Submit one bounded CPU-only architecture-reset evidence job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_SET_CLOSED_SEED:-2001}"
LOG_DIR="${SUCC_SET_CLOSED_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/set_closed_graph_rewrite_evidence_v1}"
MAIL_USER="${SUCC_SET_CLOSED_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-set-closed-v1" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_SET_CLOSED_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_SET_CLOSED_CPUS:-4}" \
  --mem="${SUCC_SET_CLOSED_MEM:-16G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-set-closed-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_set_closed_graph_rewrite_evidence.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "set_closed_evidence_job=$job_id"
echo "log=$LOG_DIR/uca-set-closed-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/set_closed_graph_rewrite_evidence_v1/seed_${SEED}/summary.json"
