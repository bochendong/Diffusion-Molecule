#!/usr/bin/env bash
# Merge the two state-viability critic arms after both terminate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_STATE_GUIDE_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_STATE_GUIDE_SEED:-1995}"
OUTPUT_ROOT="${SUCC_STATE_GUIDE_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/common_llm_state_viability_guidance_v1/seed_${SEED}}"
VALID_TERMINAL_DIR="$SHARED_PROJECT_DIR/outputs/valid_terminal_molecule_latent_jump_v1/seed_1991"

exec "$PYTHON_BIN" "$SCRIPT_DIR/merge_common_llm_state_viability_guidance.py" \
  --arm-root "$OUTPUT_ROOT" \
  --prepare-summary "$OUTPUT_ROOT/prepare/summary.json" \
  --valid-terminal-summary "$VALID_TERMINAL_DIR/summary.json" \
  --valid-terminal-candidates "$VALID_TERMINAL_DIR/evaluated_train_only_dev_candidates.csv" \
  --protocol-manifest "$SCRIPT_DIR/common_llm_state_viability_guidance_v1_preregistration.json" \
  --output "$OUTPUT_ROOT/merged/summary.json"
