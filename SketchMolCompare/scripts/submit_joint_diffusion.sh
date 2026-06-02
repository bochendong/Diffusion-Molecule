#!/usr/bin/env bash
# Submit Route B joint image+SMILES diffusion through the comparison layer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SKETCHMOL_COMPARE_SEED:-7}"
export SKETCHMOL_JOINT_RUN_NAME="${SKETCHMOL_JOINT_RUN_NAME:-sketchmol_compare_joint_diffusion_seed${SEED}}"
export SKETCHMOL_JOINT_PAIR_DIR="${SKETCHMOL_JOINT_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
export SKETCHMOL_JOINT_EPOCHS="${SKETCHMOL_JOINT_EPOCHS:-20}"
export SKETCHMOL_JOINT_BATCH_SIZE="${SKETCHMOL_JOINT_BATCH_SIZE:-128}"
export SKETCHMOL_JOINT_DIFFUSION_STEPS="${SKETCHMOL_JOINT_DIFFUSION_STEPS:-16}"
export SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT="${SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT:-1.0}"
export SKETCHMOL_JOINT_RERANK_MODE="${SKETCHMOL_JOINT_RERANK_MODE:-condition_fingerprint}"

echo "SketchMolCompare -> Route B joint diffusion"
echo "  run_name=$SKETCHMOL_JOINT_RUN_NAME"
echo "  pair_dir=$SKETCHMOL_JOINT_PAIR_DIR"
echo "  diffusion_steps=$SKETCHMOL_JOINT_DIFFUSION_STEPS"
echo "  image_loss_weight=$SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT"

bash SketchMolJointDiffusion/scripts/submit_joint_diffusion.sh "$@"
