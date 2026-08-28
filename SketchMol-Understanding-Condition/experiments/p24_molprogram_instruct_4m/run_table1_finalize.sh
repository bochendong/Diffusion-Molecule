#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$PROJECT/.." && pwd)"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
P23_OUT="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
P23="$PROJECT/experiments/p23_explicit_task_stage1_v2"
OUT="${P24_TABLE1_OUT:-$P24_OUT/eval_table1}"
MODEL_NAME="${P24_TABLE1_MODEL_NAME:-MolProgram P24 balanced refresh}"
TRAIN_DATA="${P24_TABLE1_TRAIN_DATA:-2,000,000/569,919}"
PROTOCOL="${P24_TABLE1_PROTOCOL:-p24_frozen_denovo_table1_best40_v1}"
test -f "$OUT/TABLE1_GENERATION_COMPLETE"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/merged" "$OUT/results"

merge_and_score() {
  local name="$1" conditions="$2" reference="$3" title="$4"
  "$PY" "$P23/merge_eval40.py" --expected-conditions "$conditions" \
    --prefix "$OUT/generated/${name}.raw8.csv" --extension "$OUT/generated/${name}.ranks_9_40.csv" \
    --raw-output "$OUT/merged/${name}.raw40.csv" --eval-output "$OUT/merged/${name}.eval40.csv"
  "$PY" "$REPO/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
    --eval-csv "$reference" --candidate-csv "$OUT/merged/${name}.eval40.csv" \
    --output-dir "$OUT/results/$name" --budgets 1,4,8,20,40 --report-title "$title" \
    --candidate-description "ordered frozen P24 direct-LLM candidates"
}

merge_and_score denovo_2p4p 300 \
  "$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020/data/denovo_2p4p.reference.csv" \
  "P24 balanced refresh de novo 2p-4p"
merge_and_score denovo_5p 100 \
  "$P23_OUT/eval_denovo_table1_fill/data/denovo_5p.reference.csv" \
  "P24 balanced refresh de novo 5p"
merge_and_score denovo_6p7p 40 \
  "$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data/denovo_expanded.reference.csv" \
  "P24 balanced refresh de novo 6p-7p"

"$PY" "$SCRIPT_DIR/collect_table1.py" \
  --summary "$OUT/results/denovo_2p4p/budget_sweep_summary.csv" \
  --summary "$OUT/results/denovo_5p/budget_sweep_summary.csv" \
  --summary "$OUT/results/denovo_6p7p/budget_sweep_summary.csv" \
  --output-dir "$OUT/results/table1" --model-name "$MODEL_NAME" \
  --train-data "$TRAIN_DATA" --protocol "$PROTOCOL"
touch "$OUT/TABLE1_COMPLETE"
echo "P24 Table 1 scoring complete: $OUT"
