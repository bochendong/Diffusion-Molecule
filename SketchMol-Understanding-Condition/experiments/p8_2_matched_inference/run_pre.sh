#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$PROJECT/.." && pwd)"; cd "$REPO"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; fi
PY="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; OUT="$PROJECT/outputs/p8_2_matched_inference/seed_7"; P811="$PROJECT/outputs/p8_1_1_short_transaction_r2_temperature/seed_7"; P6="$PROJECT/outputs/p6_unified_transition_policy_v1/seed_7"; DIRECT="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
mkdir -p "$OUT/data"
"$PY" "$HERE/prepare_eval.py" --denovo-eval "$DIRECT/denovo_2p7p_eval_rows.csv" --table1-eval "$P6/data/edit_table1_gate.csv" --table1-candidates "$P811/eval/edit/candidates.csv" --output-dir "$OUT/data"
touch "$OUT/PRE_COMPLETE"
