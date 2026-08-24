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
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P812_SEED:-7}"
P6_ROOT="${P812_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
ORACLE_ROOT="${P812_ORACLE_ROOT:-$PROJECT_DIR/outputs/p8_1_2_unified_transduction_v1/seed_${SEED}}"
mkdir -p "$ORACLE_ROOT/r2_validation"
"$PYTHON_BIN" "$SCRIPT_DIR/transduction_oracle.py" \
  --input-csv "$P6_ROOT/data/validation_transition_programs.csv" \
  --output-csv "$ORACLE_ROOT/r2_validation/transduction_rows.csv" \
  --summary-json "$ORACLE_ROOT/r2_validation/oracle_summary.json" \
  --variant r2_source_aligned --max-program-tokens 188
"$PYTHON_BIN" "$SCRIPT_DIR/verify_prepared_r2.py" \
  --rows "$ORACLE_ROOT/r2_validation/transduction_rows.csv" \
  --summary "$ORACLE_ROOT/r2_validation/oracle_summary.json"
