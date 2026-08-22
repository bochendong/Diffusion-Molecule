#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="${SUCC_P1_FAST_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p1_fast_hard_6p7p_seed7}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_p1_fast_hard_gate.py" \
  --eval-csv "$OUTPUT_ROOT/eval_6p7p_128_each.csv" \
  --sft-candidates "$OUTPUT_ROOT/sft/raw_candidates_n20.csv" \
  --group-rl-candidates "$OUTPUT_ROOT/group_rl/raw_candidates_n20.csv" \
  --output-dir "$OUTPUT_ROOT/final" --bootstrap-resamples 5000 --seed 20260823

echo "P1 fast report=$OUTPUT_ROOT/final/report.md"
