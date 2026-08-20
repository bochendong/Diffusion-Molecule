#!/usr/bin/env bash
# Submit one short 20GB MIG representation kill test; no downstream generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_PROPERTY_BASIS_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/property_factorized_language_graph_basis_v1}"
MAIL_USER="${SUCC_PROPERTY_BASIS_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_PROPERTY_BASIS_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_PROPERTY_BASIS_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

job_id="$(sbatch --parsable \
  --job-name=uca-property-basis-v1 \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=4 --mem=20G \
  --gres="gpu:$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-property-basis-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_property_factorized_language_graph_basis_v1.sh'")"

echo "job_id=$job_id"
echo "gpu=$GPU_REQUEST"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/property_factorized_language_graph_basis_v1/seed_2051/summary.json"
