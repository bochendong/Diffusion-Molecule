#!/usr/bin/env bash
# Submit Route A token diffusion with SELFIES internals for higher valid decode rate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SKETCHMOL_COMPARE_SEED:-7}"
export SKETCHMOL_TOKEN_RUN_NAME="${SKETCHMOL_TOKEN_RUN_NAME:-sketchmol_compare_token_diffusion_selfies_seed${SEED}}"
export SKETCHMOL_TOKEN_TOKENIZATION="${SKETCHMOL_TOKEN_TOKENIZATION:-selfies}"
export SKETCHMOL_TOKEN_DECODE_LENGTH_MODE="${SKETCHMOL_TOKEN_DECODE_LENGTH_MODE:-train_median}"
export SKETCHMOL_TOKEN_MIN_DECODE_TOKENS="${SKETCHMOL_TOKEN_MIN_DECODE_TOKENS:-4}"

echo "SketchMolCompare -> Route A SELFIES token diffusion"
echo "  run_name=$SKETCHMOL_TOKEN_RUN_NAME"
echo "  tokenization=$SKETCHMOL_TOKEN_TOKENIZATION"
echo "  decode_length_mode=$SKETCHMOL_TOKEN_DECODE_LENGTH_MODE"

bash SketchMolCompare/scripts/submit_token_diffusion.sh "$@"
