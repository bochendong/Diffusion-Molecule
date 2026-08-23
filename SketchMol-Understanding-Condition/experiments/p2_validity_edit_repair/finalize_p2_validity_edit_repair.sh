#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${P2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p2_validity_edit_repair_seed7}"
EDIT_ROOT="${P2_EDIT_ROOT:-$PROJECT_DIR/outputs/p1_source_consistency_validity_pilot_v1/seed_7}"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_p2_validity_edit_repair.py" \
  --output-root "$OUTPUT_ROOT" \
  --edit-metrics-csv "$EDIT_ROOT/p1_source_consistency_validity_metrics.csv" \
  --budgets "${P2_BUDGETS:-1,8,20}"

echo "P2 report: $OUTPUT_ROOT/final/p2_report.md"
