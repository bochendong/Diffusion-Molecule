#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P34_SCRIPT_DIR:?P34_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P34_OUTPUT_ROOT:-$PROJECT/outputs/p34_hard_boundary_edit_rl/seed_34001}"
PY="${P34_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_gate.py" --eval-root "$OUT/eval" --output-dir "$OUT/small_gate"
