#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P812_SEED:-7}"
P6_ROOT="${P812_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
OUTPUT_ROOT="${P812_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p8_1_2_unified_transduction_v1/seed_${SEED}}"
INPUT_CSV="${P812_INPUT_CSV:-$P6_ROOT/data/train_transition_programs.csv}"

[[ -f "$INPUT_CSV" ]] || { echo "ERROR: missing shared P6 rows: $INPUT_CSV" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/r1" "$OUTPUT_ROOT/r2"

"$PYTHON_BIN" "$SCRIPT_DIR/transduction_oracle.py" \
  --input-csv "$INPUT_CSV" \
  --output-csv "$OUTPUT_ROOT/r1/transduction_rows.csv" \
  --summary-json "$OUTPUT_ROOT/r1/oracle_summary.json" \
  --variant r1_canonical --max-program-tokens 188

# R2 changes exactly one factor: target atom serialization is aligned to the
# source MCS before SELFIES encoding. The language/interpreter/data remain fixed.
"$PYTHON_BIN" "$SCRIPT_DIR/transduction_oracle.py" \
  --input-csv "$INPUT_CSV" \
  --output-csv "$OUTPUT_ROOT/r2/transduction_rows.csv" \
  --summary-json "$OUTPUT_ROOT/r2/oracle_summary.json" \
  --variant r2_source_aligned --max-program-tokens 188

touch "$OUTPUT_ROOT/COMPLETE"
echo "P8.1.2 R1/R2 oracle complete: $OUTPUT_ROOT"
