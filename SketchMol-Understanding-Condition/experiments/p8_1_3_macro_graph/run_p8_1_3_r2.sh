#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
SEED="${P8_1_3_SEED:-7}"
P6_ROOT="${P8_1_3_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
R1_DIR="${P8_1_3_R1_OUTPUT_DIR:-$PROJECT_DIR/outputs/p8_1_3_macro_graph/r1/seed_${SEED}}"
OUTPUT_DIR="${P8_1_3_R2_OUTPUT_DIR:-$PROJECT_DIR/outputs/p8_1_3_macro_graph/r2/seed_${SEED}}"
[[ -f "$R1_DIR/summary.json" ]] || { echo "missing completed R1 summary" >&2; exit 2; }
# R1's expected failure is whole-region vocabulary OOV.  R2 changes only the
# payload factorization; split, seed, transaction and interpreter are fixed.
python3 "$SCRIPT_DIR/audit_macro_transaction_gate.py" \
  --input-csv "$P6_ROOT/data/train_transition_programs.csv" \
  --output-dir "$OUTPUT_DIR" --payload-factor brics --seed "$SEED"
touch "$OUTPUT_DIR/COMPLETE"
