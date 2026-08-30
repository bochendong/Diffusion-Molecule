#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P324_SCRIPT_DIR:?P324_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P324_OUTPUT_ROOT:-$PROJECT/outputs/p32_4_edit_specialist_source_rl/seed_32401}"
PY="${P324_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_gate.py" --eval-root "$OUT/eval" --output-dir "$OUT/small_gate"
