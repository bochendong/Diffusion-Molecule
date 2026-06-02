#!/usr/bin/env bash
# Submit Route A token diffusion through the comparison layer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SKETCHMOL_COMPARE_SEED:-7}"
export SKETCHMOL_TOKEN_RUN_NAME="${SKETCHMOL_TOKEN_RUN_NAME:-sketchmol_compare_token_diffusion_seed${SEED}}"
export SKETCHMOL_TOKEN_PAIR_DIR="${SKETCHMOL_TOKEN_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
export SKETCHMOL_TOKEN_EPOCHS="${SKETCHMOL_TOKEN_EPOCHS:-20}"
export SKETCHMOL_TOKEN_BATCH_SIZE="${SKETCHMOL_TOKEN_BATCH_SIZE:-128}"
export SKETCHMOL_TOKEN_DIFFUSION_STEPS="${SKETCHMOL_TOKEN_DIFFUSION_STEPS:-16}"
export SKETCHMOL_TOKEN_RERANK_MODE="${SKETCHMOL_TOKEN_RERANK_MODE:-condition_fingerprint}"

echo "SketchMolCompare -> Route A token diffusion"
echo "  run_name=$SKETCHMOL_TOKEN_RUN_NAME"
echo "  pair_dir=$SKETCHMOL_TOKEN_PAIR_DIR"
echo "  diffusion_steps=$SKETCHMOL_TOKEN_DIFFUSION_STEPS"

bash SketchMolTokenDiffusion/scripts/submit_masked_token_diffusion.sh "$@"
