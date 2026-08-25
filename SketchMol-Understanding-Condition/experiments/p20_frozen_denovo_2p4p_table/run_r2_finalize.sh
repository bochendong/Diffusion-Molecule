#!/usr/bin/env bash
set -euo pipefail
HERE="${P20_SCRIPT_DIR:?P20_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$HERE/../.." && pwd)"
REPO="$(cd "$PROJECT/.." && pwd)"
PY="${P20_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P20_OUTPUT_ROOT:-$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020}"
test -f "$OUT/r2/P17_EXTENSION_COMPLETE"
test -f "$OUT/r2/P18_EXTENSION_COMPLETE"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
for model in p17 p18; do
  "$PY" "$HERE/merge_r2.py" --audit "$OUT/r2/audit_before_extension.json" --model "$model" \
    --prefix "$OUT/generated/$model/denovo.raw.csv" --extension "$OUT/r2/generated/$model/ranks_9_40.csv" \
    --raw-output "$OUT/r2/generated/$model/denovo.raw40.csv" --eval-output "$OUT/r2/generated/$model/denovo.eval40.csv"
  "$PY" "$REPO/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
    --eval-csv "$OUT/data/denovo_2p4p.reference.csv" --candidate-csv "$OUT/r2/generated/$model/denovo.eval40.csv" \
    --output-dir "$OUT/r2/results/budget_sweep/$model" --budgets 1,4,8,20,40 \
    --report-title "P20 R2 Frozen ${model^^} De Novo 2p-4p Fair @40 Pilot" \
    --candidate-description "ordered frozen direct-LLM candidates"
done
"$PY" "$HERE/aggregate_r2.py" \
  --p17-summary "$OUT/r2/results/budget_sweep/p17/budget_sweep_summary.csv" \
  --p18-summary "$OUT/r2/results/budget_sweep/p18/budget_sweep_summary.csv" \
  --output-dir "$OUT/r2/results"
touch "$OUT/r2/R2_FINAL_COMPLETE"
