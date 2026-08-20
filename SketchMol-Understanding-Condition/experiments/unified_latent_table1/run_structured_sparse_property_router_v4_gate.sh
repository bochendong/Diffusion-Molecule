#!/usr/bin/env bash
# Apply the separate scientific gate after all four execution arms complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_SPARSE_ROUTER_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ARMS_ROOT="${SUCC_SPARSE_ROUTER_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/structured_sparse_property_router_v4/seed_2061}"
GATE_OUTPUT_DIR="${SUCC_SPARSE_ROUTER_GATE_OUTPUT_DIR:-$ARMS_ROOT/gate}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/gate_structured_sparse_property_router_v4.py" \
  --protocol-manifest "$SCRIPT_DIR/structured_sparse_property_router_v4_preregistration.json" \
  --arms-root "$ARMS_ROOT" \
  --output-dir "$GATE_OUTPUT_DIR"
