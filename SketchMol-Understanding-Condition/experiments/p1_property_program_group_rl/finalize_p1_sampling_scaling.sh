#!/usr/bin/env bash
# Finalize the four frozen P1 candidate pools into paired scaling tables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${SUCC_P1_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p1_property_program_group_rl_seed7}"
TWO_P_BASE="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
OOD_BASE="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_mixed_condition"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_p1_sampling_scaling.py" \
  --two-p-seven-p-eval-csv "$TWO_P_BASE/denovo_2p7p_eval_rows.csv" \
  --ood-eval-csv "$OOD_BASE/denovo_ood_eval_rows.csv" \
  --two-p-seven-p-sft-candidates "$OUTPUT_ROOT/two_p_to_seven_p_sft/raw_candidates_n256.csv" \
  --two-p-seven-p-group-rl-candidates "$OUTPUT_ROOT/two_p_to_seven_p_group_rl/raw_candidates_n256.csv" \
  --ood-sft-candidates "$OUTPUT_ROOT/ood_sft/raw_candidates_n256.csv" \
  --ood-group-rl-candidates "$OUTPUT_ROOT/ood_group_rl/raw_candidates_n256.csv" \
  --output-dir "$OUTPUT_ROOT/final" \
  --budgets "1,4,8,20,32,64,128,256" \
  --bootstrap-resamples "${SUCC_P1_BOOTSTRAP_RESAMPLES:-5000}" \
  --seed 20260822 \
  --protocol "$SCRIPT_DIR/p1_property_program_group_rl_preregistration.json"

echo "P1 final report=$OUTPUT_ROOT/final/p1_report.md"
echo "P1 gate=$OUTPUT_ROOT/final/p1_gate.json"
