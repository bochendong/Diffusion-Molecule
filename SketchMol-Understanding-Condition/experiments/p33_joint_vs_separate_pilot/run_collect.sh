#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P33_SCRIPT_DIR:?P33_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P33_OUTPUT_ROOT:-$PROJECT/outputs/p33_joint_vs_separate_pilot/seed_33001}"
PY="${P33_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_pilot.py" --output-root "$OUT"
