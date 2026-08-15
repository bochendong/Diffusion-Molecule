#!/usr/bin/env bash
# Submit the B18 continuous constraint transport pilot on one 10 GB H100 MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_CONTINUOUS_TRANSPORT_SEED:-1751}"
LOG_DIR="${SUCC_CONTINUOUS_TRANSPORT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/continuous_constraint_transport_v18}"
MAIL_USER="${SUCC_CONTINUOUS_TRANSPORT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-cont-flow-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_CONTINUOUS_TRANSPORT_TIME:-00:10:00}" \
  --cpus-per-task=1 \
  --mem=4G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-cont-flow-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_continuous_constraint_transport_pilot.sh'")"

echo "$submission"
echo "continuous_constraint_transport_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/continuous_constraint_transport_v18/seed_${SEED}/summary.json"
