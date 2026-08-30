#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P321_SCRIPT_DIR:?P321_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P321_OUTPUT_ROOT:-$PROJECT/outputs/p32_1_verifier_routed_residual_rl/seed_32101}"
PY="${P321_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_residual_gate.py" \
  --eval-root "$OUT/eval" --output-dir "$OUT/joint_gate"
