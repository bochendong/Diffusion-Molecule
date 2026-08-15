#!/usr/bin/env bash
# Submit the bounded B20 discrete graph diffusion decoder pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_DISCRETE_DIFFUSION_SEED:-1753}"
LOG_DIR="${SUCC_DISCRETE_DIFFUSION_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/discrete_graph_diffusion_decoder_v20}"
MAIL_USER="${SUCC_DISCRETE_DIFFUSION_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-discrete-diff-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_DISCRETE_DIFFUSION_TIME:-00:20:00}" \
  --cpus-per-task=1 \
  --mem=8G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-discrete-diff-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_discrete_graph_diffusion_decoder_pilot.sh'")"

echo "$submission"
echo "discrete_graph_diffusion_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "pilot_summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/discrete_graph_diffusion_decoder_v20/seed_${SEED}/summary.json"
