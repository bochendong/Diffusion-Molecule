#!/usr/bin/env bash
# Submit Route B with SELFIES tokens, a shared latent, and CLIP-style alignment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SKETCHMOL_COMPARE_SEED:-7}"
export SKETCHMOL_JOINT_RUN_NAME="${SKETCHMOL_JOINT_RUN_NAME:-sketchmol_compare_joint_selfies_latent_clip_seed${SEED}}"
export SKETCHMOL_JOINT_TOKENIZATION="${SKETCHMOL_JOINT_TOKENIZATION:-selfies}"
export SKETCHMOL_JOINT_LATENT_DIM="${SKETCHMOL_JOINT_LATENT_DIM:-128}"
export SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT="${SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT:-1.0}"
export SKETCHMOL_JOINT_CLIP_LOSS_WEIGHT="${SKETCHMOL_JOINT_CLIP_LOSS_WEIGHT:-0.1}"
export SKETCHMOL_JOINT_CLIP_TEMPERATURE="${SKETCHMOL_JOINT_CLIP_TEMPERATURE:-0.07}"
export SKETCHMOL_JOINT_DECODE_LENGTH_MODE="${SKETCHMOL_JOINT_DECODE_LENGTH_MODE:-train_median}"
export SKETCHMOL_JOINT_MIN_DECODE_TOKENS="${SKETCHMOL_JOINT_MIN_DECODE_TOKENS:-4}"

echo "SketchMolCompare -> Route B SELFIES latent-CLIP joint diffusion"
echo "  run_name=$SKETCHMOL_JOINT_RUN_NAME"
echo "  tokenization=$SKETCHMOL_JOINT_TOKENIZATION"
echo "  latent_dim=$SKETCHMOL_JOINT_LATENT_DIM"
echo "  image_loss_weight=$SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT"
echo "  clip_loss_weight=$SKETCHMOL_JOINT_CLIP_LOSS_WEIGHT"
echo "  decode_length_mode=$SKETCHMOL_JOINT_DECODE_LENGTH_MODE"

bash SketchMolCompare/scripts/submit_joint_diffusion.sh "$@"
