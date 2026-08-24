#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$PROJECT/.." && pwd)"; cd "$REPO"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; fi
PY="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; OUT="$PROJECT/outputs/p8_2_matched_inference/seed_7/table1"; P811="$PROJECT/outputs/p8_1_1_short_transaction_r2_temperature/seed_7"; REF="$PROJECT/outputs/p6_unified_transition_policy_v1/seed_7/data/edit_table1_gate.csv"; CAND="$P811/eval/edit/candidates.csv"
for k in 1 8 20; do "$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" --reference "$REF" --candidates "$CAND" --output-dir "$OUT/any$k" --candidate-limit "$k" --model-name "P8.2-P8.1.1-R2-any$k" --task-filter table1 --missing-oracle-policy fail; done
"$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" --reference "$REF" --candidates "$CAND" --output-dir "$OUT/candidate20" --candidate-limit 20 --aggregation candidate --model-name P8.2-P8.1.1-R2-candidate20 --task-filter table1 --missing-oracle-policy fail
touch "$OUT/COMPLETE"
