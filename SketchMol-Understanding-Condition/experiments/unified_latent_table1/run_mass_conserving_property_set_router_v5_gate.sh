#!/usr/bin/env bash
# Apply the distinct V5 science gate after all four execution arms complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_SET_ROUTER_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ARMS_ROOT="${SUCC_SET_ROUTER_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/mass_conserving_property_set_router_v5/seed_2071}"
GATE_OUTPUT_DIR="${SUCC_SET_ROUTER_GATE_OUTPUT_DIR:-$ARMS_ROOT/gate}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/gate_mass_conserving_property_set_router_v5.py" \
  --protocol-manifest "$SCRIPT_DIR/mass_conserving_property_set_router_v5_preregistration.json" \
  --arms-root "$ARMS_ROOT" \
  --output-dir "$GATE_OUTPUT_DIR"
