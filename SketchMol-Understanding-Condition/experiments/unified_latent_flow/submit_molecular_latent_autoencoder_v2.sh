#!/usr/bin/env bash
# Submit the bounded Stage-A molecular latent autoencoder gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_MOL_AE_SEED:-1717}"
LOG_DIR="${SUCC_MOL_AE_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/molecular_latent_autoencoder_v2}"
MAIL_USER="${SUCC_MOL_AE_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-mol-ae-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_MOL_AE_TIME:-01:00:00}" \
  --cpus-per-task="${SUCC_MOL_AE_CPUS:-4}" \
  --mem="${SUCC_MOL_AE_MEM:-32G}" \
  --gres="${SUCC_MOL_AE_GRES:-gpu:nvidia_h100_80gb_hbm3_2g.20gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-mol-ae-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_molecular_latent_autoencoder_v2.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "molecular_latent_autoencoder_job=$job_id"
echo "log=$LOG_DIR/uca-mol-ae-s${SEED}-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/molecular_latent_autoencoder_v2/seed_${SEED}/summary.json"
