#!/usr/bin/env bash
# Submit the broader SketchSMILES candidate-generation matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting SketchSMILES candidate-generation matrix"
echo "  scripts_dir=$SCRIPT_DIR"

bash "$SCRIPT_DIR/submit_sketchsmiles_5a6_aug_sweep.sh" "$@"
bash "$SCRIPT_DIR/submit_sketchsmiles_5a4_beam32.sh" "$@"
bash "$SCRIPT_DIR/submit_sketchsmiles_5a4_sample64.sh" "$@"
bash "$SCRIPT_DIR/submit_sketchsmiles_5a4_large_transformer.sh" "$@"
bash "$SCRIPT_DIR/submit_sketchsmiles_5d_strong_fingerprint.sh" "$@"
