#!/usr/bin/env bash
# Audit existing teacher pools and materialize strict-first trajectories without GPU work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-python3}"
CONFIG_JSON="${SUCC_UCA_CONFIG_JSON:?Set SUCC_UCA_CONFIG_JSON to the run manifest}"
OUTPUT_DIR="${SUCC_UCA_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_constraint_agent_teacher_audit_v1}"

"$PYTHON_BIN" "$SCRIPT_DIR/audit_candidate_pools.py" \
  --config-json "$CONFIG_JSON" \
  --output-dir "$OUTPUT_DIR/audit"

"$PYTHON_BIN" "$SCRIPT_DIR/build_verifier_trajectories.py" \
  --config-json "$CONFIG_JSON" \
  --output-dir "$OUTPUT_DIR/trajectories"

echo "Unified constraint teacher audit ready: $OUTPUT_DIR"
