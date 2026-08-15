#!/usr/bin/env bash
# Submit the B28 zero-training CPU quantizer kill test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_ENERGY_TILTED_VQ_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/energy_tilted_vq_fragment_sampling_v28}"
MAIL_USER="${SUCC_ENERGY_TILTED_VQ_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-energy-vq-v28" \
  --account=def-hup-ab_cpu \
  --time="00:15:00" \
  --cpus-per-task=4 \
  --mem=12G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-energy-vq-v28-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_energy_tilted_vq_fragment_sampling.sh'")"

echo "$submission"
echo "energy_tilted_vq_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/energy_tilted_vq_fragment_sampling_v28/seed_1901/summary.json"
