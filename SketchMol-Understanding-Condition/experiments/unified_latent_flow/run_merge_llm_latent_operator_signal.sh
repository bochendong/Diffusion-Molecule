#!/usr/bin/env bash
# Merge the four frozen common-LLM latent-operator signal arms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_LLM_LATENT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_LLM_LATENT_SEED:-1993}"
OUTPUT_ROOT="${SUCC_LLM_LATENT_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/common_llm_latent_operator_signal_v1/seed_${SEED}}"
VALID_TERMINAL_SUMMARY="$SHARED_PROJECT_DIR/outputs/valid_terminal_molecule_latent_jump_v1/seed_1991/summary.json"
PREREGISTRATION="$SCRIPT_DIR/llm_latent_operator_signal_v1_preregistration.json"

mkdir -p "$OUTPUT_ROOT/merged"
exec "$PYTHON_BIN" "$SCRIPT_DIR/merge_llm_latent_operator_signal.py" \
  --arm-root "$OUTPUT_ROOT" \
  --valid-terminal-summary "$VALID_TERMINAL_SUMMARY" \
  --protocol-manifest "$PREREGISTRATION" \
  --output "$OUTPUT_ROOT/merged/summary.json"
