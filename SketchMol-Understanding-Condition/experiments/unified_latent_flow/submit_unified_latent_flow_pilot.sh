#!/usr/bin/env bash
# Submit one bounded H100 20GB MIG latent-flow pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_LATENT_FLOW_SEED:-1715}"
LOG_DIR="${SUCC_LATENT_FLOW_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/unified_latent_flow_pilot_v1}"
MAIL_USER="${SUCC_LATENT_FLOW_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-latent-flow-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_LATENT_FLOW_TIME:-02:00:00}" \
  --cpus-per-task="${SUCC_LATENT_FLOW_CPUS:-4}" \
  --mem="${SUCC_LATENT_FLOW_MEM:-32G}" \
  --gres="${SUCC_LATENT_FLOW_GRES:-gpu:nvidia_h100_80gb_hbm3_2g.20gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-latent-flow-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_unified_latent_flow_pilot.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "latent_flow_job=$job_id"
echo "log=$LOG_DIR/uca-latent-flow-s${SEED}-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/unified_latent_flow_pilot_v1/seed_${SEED}/summary.json"
