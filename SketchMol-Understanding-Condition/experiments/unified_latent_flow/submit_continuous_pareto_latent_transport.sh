#!/usr/bin/env bash
# Submit the bounded single-seed B34 CPU signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_CONTINUOUS_PARETO_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/continuous_pareto_latent_transport_v34}"
MAIL_USER="${SUCC_CONTINUOUS_PARETO_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-cont-pareto-v34" \
  --account=def-hup-ab_cpu \
  --time="00:45:00" \
  --cpus-per-task=4 \
  --mem=20G \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-cont-pareto-v34-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_continuous_pareto_latent_transport.sh'")"

echo "$submission"
echo "continuous_pareto_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/continuous_pareto_latent_transport_v34/seed_1961/summary.json"
