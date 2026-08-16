#!/usr/bin/env bash
# Submit one bounded CPU-only universal continuous graph-delta signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_UNIVERSAL_CONTINUOUS_SEED:-2015}"
LOG_DIR="${SUCC_UNIVERSAL_CONTINUOUS_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/universal_continuous_graph_delta_flow_v1}"
MAIL_USER="${SUCC_UNIVERSAL_CONTINUOUS_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-univ-cont-v1" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_UNIVERSAL_CONTINUOUS_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_UNIVERSAL_CONTINUOUS_CPUS:-4}" \
  --mem="${SUCC_UNIVERSAL_CONTINUOUS_MEM:-8G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-univ-cont-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_universal_continuous_graph_delta_flow.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "universal_continuous_job=$job_id"
echo "log=$LOG_DIR/uca-univ-cont-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/universal_continuous_graph_delta_flow_v1/seed_${SEED}/summary.json"
