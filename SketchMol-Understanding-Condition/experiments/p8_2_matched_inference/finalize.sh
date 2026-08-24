#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$PROJECT/.." && pwd)"; cd "$REPO"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; fi
PY="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; OUT="$PROJECT/outputs/p8_2_matched_inference/seed_7"; P811R1="$PROJECT/outputs/p8_1_1_short_transaction_r1/seed_7"; P811R2="$PROJECT/outputs/p8_1_1_short_transaction_r2_temperature/seed_7"; DIRECT="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
CAND=(); SUM=(); for pc in 2 3 4 5 6 7; do CAND+=("$OUT/denovo/pc$pc/candidates.csv"); SUM+=("$OUT/denovo/pc$pc/sampling_summary.json"); done
"$PY" "$HERE/evaluate_complexity.py" --eval-csv "$DIRECT/denovo_2p7p_eval_rows.csv" --candidates "${CAND[@]}" --output-dir "$OUT/denovo" --budgets 1,4,8,20
"$PY" "$HERE/audit_matched.py" --checkpoint "$P811R1/policy/umtp_graph_action_policy.pt" --p811-audit "$P811R2/final_audit.json" --edit-candidates "$P811R2/eval/edit/candidates.csv" --edit-sampling-summary "$P811R2/eval/edit/sampling_summary.json" --table1-output "$OUT/table1" --support-audit "$OUT/data/support_audit.json" --denovo-candidates "$OUT/denovo/candidates_merged.csv" --denovo-summaries "${SUM[@]}" --output "$OUT/matched_audit.json"
touch "$OUT/COMPLETE"
