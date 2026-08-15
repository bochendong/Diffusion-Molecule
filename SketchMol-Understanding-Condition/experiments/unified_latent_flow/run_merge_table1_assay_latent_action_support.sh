#!/usr/bin/env bash
# Merge the eight completed B30-r1 support shards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_ASSAY_SUPPORT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${SUCC_ASSAY_SUPPORT_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/table1_assay_latent_action_support_v30_r1/seed_1922}"
PREREGISTRATION="$SCRIPT_DIR/table1_assay_latent_action_support_v30_r1_preregistration.json"

[[ ! -f "$OUTPUT_ROOT/merged/summary.json" ]] || {
  echo "ERROR: completed B30 merged result exists: $OUTPUT_ROOT/merged/summary.json" >&2
  exit 2
}

export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/merge_table1_assay_latent_action_support.py" \
  --shard-root "$OUTPUT_ROOT/shards" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_ROOT/merged"
