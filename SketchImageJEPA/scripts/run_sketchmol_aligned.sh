#!/usr/bin/env bash
# Run SketchImage-JEPA with defaults chosen to line up with SketchMol-style
# comparison settings rather than minimal smoke-test settings.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export SKETCHIMAGE_PRESET="${SKETCHIMAGE_PRESET:-sketchmol_aligned}"
export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-sketchimage_sketchmol_aligned_$(date +%Y%m%d_%H%M%S)}"
export SKETCHIMAGE_FEATURE_DIM="${SKETCHIMAGE_FEATURE_DIM:-256}"
export SKETCHIMAGE_LATENT_DIM="${SKETCHIMAGE_LATENT_DIM:-4096}"
export SKETCHIMAGE_TOP_K="${SKETCHIMAGE_TOP_K:-8}"
export SKETCHIMAGE_TRAIN_FRACTION="${SKETCHIMAGE_TRAIN_FRACTION:-0.8}"
export SKETCHIMAGE_SEED="${SKETCHIMAGE_SEED:-7}"
export SKETCHIMAGE_RENDER_IMAGE_CONTEXT="${SKETCHIMAGE_RENDER_IMAGE_CONTEXT:-1}"

echo "SketchMol-aligned defaults:"
echo "  condition/context dim: 256"
echo "  latent shape: 32x32x4 = 4096"
echo "  candidates per condition: 8"
echo "  image size reference: 256"
echo "  SketchMol DDIM reference: 250 steps, eta=1.0, scale=2, scale_pro=4"
echo

bash scripts/run_one_click.sh
