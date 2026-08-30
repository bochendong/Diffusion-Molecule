#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P32_SCRIPT_DIR:?P32_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P32_OUTPUT_ROOT:-$PROJECT/outputs/p32_unified_graph_repair_rl/seed_32001}"
PY="${P32_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_gate.py" --eval-root "$OUT/eval" --output-dir "$OUT/joint_gate"
