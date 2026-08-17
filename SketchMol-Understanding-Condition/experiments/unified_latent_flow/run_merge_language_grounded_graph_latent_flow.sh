#!/usr/bin/env bash
# Merge the two language-grounded transport arms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_LANG_FLOW_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_LANG_FLOW_SEED:-2001}"
OUTPUT_ROOT="${SUCC_LANG_FLOW_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/language_grounded_graph_latent_flow_v1/seed_${SEED}}"
VALID_TERMINAL_DIR="$SHARED_PROJECT_DIR/outputs/valid_terminal_molecule_latent_jump_v1/seed_1991"

exec "$PYTHON_BIN" "$SCRIPT_DIR/merge_language_grounded_graph_latent_flow.py" \
  --arm-root "$OUTPUT_ROOT" \
  --valid-terminal-summary "$VALID_TERMINAL_DIR/summary.json" \
  --protocol-manifest "$SCRIPT_DIR/language_grounded_graph_latent_flow_v1_preregistration.json" \
  --output "$OUTPUT_ROOT/merged/summary.json"
