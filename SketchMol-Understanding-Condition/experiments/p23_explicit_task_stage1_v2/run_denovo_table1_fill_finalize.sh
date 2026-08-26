#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$PROJECT/.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P23_DENOVO_FILL_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/eval_denovo_table1_fill}"
P19="$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919"
P20="$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/results/budget_sweep/p18/budget_sweep_summary.csv"
ALIGNED_PAPER="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/eval_corrected_prompts/results"
test -f "$OUT/DENOVO_FILL_GENERATED"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/merged" "$OUT/results"

"$PY" "$SCRIPT_DIR/merge_eval40.py" --expected-conditions 100 \
  --prefix "$OUT/generated/aligned/denovo_5p.raw8.csv" --extension "$OUT/generated/aligned/denovo_5p.ranks_9_40.csv" \
  --raw-output "$OUT/merged/aligned_5p.raw40.csv" --eval-output "$OUT/merged/aligned_5p.eval40.csv"
"$PY" "$SCRIPT_DIR/merge_eval40.py" --expected-conditions 100 \
  --prefix "$OUT/generated/legacy/denovo_5p.raw8.csv" --extension "$OUT/generated/legacy/denovo_5p.ranks_9_40.csv" \
  --raw-output "$OUT/merged/legacy_5p.raw40.csv" --eval-output "$OUT/merged/legacy_5p.eval40.csv"
"$PY" "$SCRIPT_DIR/merge_eval40.py" --expected-conditions 40 \
  --prefix "$P19/generated/p18/denovo.raw.csv" --extension "$OUT/generated/legacy/denovo_6p7p.ranks_9_40.csv" \
  --raw-output "$OUT/merged/legacy_6p7p.raw40.csv" --eval-output "$OUT/merged/legacy_6p7p.eval40.csv"

for model in aligned legacy; do
  "$PY" "$REPO/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
    --eval-csv "$OUT/data/denovo_5p.reference.csv" --candidate-csv "$OUT/merged/${model}_5p.eval40.csv" \
    --output-dir "$OUT/results/${model}_5p" --budgets 1,4,8,20,40 \
    --report-title "P23 paper Table 1 ${model} 5p fill" --candidate-description "ordered frozen direct-LLM candidates"
done
"$PY" "$REPO/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
  --eval-csv "$P19/data/denovo_expanded.reference.csv" --candidate-csv "$OUT/merged/legacy_6p7p.eval40.csv" \
  --output-dir "$OUT/results/legacy_6p7p" --budgets 1,4,8,20,40 \
  --report-title "P18 legacy 160/160 de novo 6p-7p paper fill" --candidate-description "ordered frozen direct-LLM candidates"

"$PY" "$SCRIPT_DIR/collect_denovo_table1_fill.py" \
  --legacy-2p4p "$P20" --legacy-5p "$OUT/results/legacy_5p/budget_sweep_summary.csv" \
  --legacy-6p7p "$OUT/results/legacy_6p7p/budget_sweep_summary.csv" \
  --aligned-2p4p "$ALIGNED_PAPER/denovo_2p4p/budget_sweep_summary.csv" \
  --aligned-5p "$OUT/results/aligned_5p/budget_sweep_summary.csv" \
  --aligned-6p7p "$ALIGNED_PAPER/denovo_6p7p/budget_sweep_summary.csv" \
  --output-dir "$OUT/results/table1"
touch "$OUT/DENOVO_FILL_COMPLETE"
