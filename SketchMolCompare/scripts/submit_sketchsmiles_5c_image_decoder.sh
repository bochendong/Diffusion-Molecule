#!/usr/bin/env bash
# Submit the image-conditioned OCR-free SketchSMILES decoder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SKETCHSMILES_DIR="$REPO_ROOT/SketchSMILES"

if [[ ! -d "$SKETCHSMILES_DIR" ]]; then
  echo "ERROR: SketchSMILES folder not found at $SKETCHSMILES_DIR" >&2
  exit 2
fi

cd "$SKETCHSMILES_DIR"

SEED="${SKETCHMOL_COMPARE_SEED:-${SKETCHSMILES_SEED:-7}}"
export SKETCHSMILES_SEED="$SEED"
export SKETCHSMILES_RUN_NAME="${SKETCHSMILES_RUN_NAME:-sketchmol_compare_phase5c_seed${SEED}}"
export SKETCHSMILES_PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
export SKETCHSMILES_EPOCHS="${SKETCHSMILES_EPOCHS:-20}"
export SKETCHSMILES_BATCH_SIZE="${SKETCHSMILES_BATCH_SIZE:-128}"
export SKETCHSMILES_BEAM_SIZE="${SKETCHSMILES_BEAM_SIZE:-8}"
export SKETCHSMILES_IMAGE_SIZE="${SKETCHSMILES_IMAGE_SIZE:-128}"
export SKETCHSMILES_GPU_PROFILE="${SKETCHSMILES_GPU_PROFILE:-h100_10gb_mig}"
export SKETCHSMILES_SLURM_TIME="${SKETCHSMILES_SLURM_TIME:-10:00:00}"
export SKETCHSMILES_SLURM_MEM="${SKETCHSMILES_SLURM_MEM:-32G}"

echo "SketchMolCompare -> SketchSMILES Phase 5C"
echo "  cwd=$PWD"
echo "  run_name=$SKETCHSMILES_RUN_NAME"
echo "  pair_dir=$SKETCHSMILES_PAIR_DIR"
echo "  image_size=$SKETCHSMILES_IMAGE_SIZE"

bash scripts/submit_phase5c_image_smiles_decoder.sh "$@"
