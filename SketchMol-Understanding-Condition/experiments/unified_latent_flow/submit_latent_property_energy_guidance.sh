#!/usr/bin/env bash
# Submit the B27 single-seed CPU kill test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_LATENT_ENERGY_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/latent_property_energy_guidance_v27}"
MAIL_USER="${SUCC_LATENT_ENERGY_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-latent-energy-v27" \
  --account=def-hup-ab_cpu \
  --time="00:30:00" \
  --cpus-per-task=4 \
  --mem=12G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-latent-energy-v27-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_latent_property_energy_guidance.sh'")"

echo "$submission"
echo "latent_property_energy_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/latent_property_energy_guidance_v27/seed_1891/summary.json"
