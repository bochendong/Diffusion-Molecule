#!/usr/bin/env bash
# Submit the B29 zero-training CPU Table1 transfer kill test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_TABLE1_LATENT_TRANSFER_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/table1_energy_tilted_latent_transfer_v29}"
MAIL_USER="${SUCC_TABLE1_LATENT_TRANSFER_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-table1-latent-v29" \
  --account=def-hup-ab_cpu \
  --time="00:30:00" \
  --cpus-per-task=4 \
  --mem=12G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-table1-latent-v29-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_table1_energy_tilted_latent_transfer.sh'")"

echo "$submission"
echo "table1_latent_transfer_job=$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/table1_energy_tilted_latent_transfer_v29/seed_1911/summary.json"
