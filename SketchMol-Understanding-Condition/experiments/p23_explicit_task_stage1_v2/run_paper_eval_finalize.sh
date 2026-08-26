#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$PROJECT/.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
OUT="${P23_EVAL_OUTPUT_ROOT:-$TRAIN_OUT/eval_paper_protocol}"
OLD_EVAL="$TRAIN_OUT/eval_corrected_prompts"

test -f "$OUT/PAPER_GENERATION_COMPLETE"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/merged" "$OUT/results"

"$PY" "$SCRIPT_DIR/merge_eval40.py" --expected-conditions 300 \
  --prefix "$OLD_EVAL/generated/denovo_2p4p_p20.raw_k8.csv" --extension "$OUT/generated/denovo_2p4p.ranks_9_40.csv" \
  --raw-output "$OUT/merged/denovo_2p4p.raw40.csv" --eval-output "$OUT/merged/denovo_2p4p.eval40.csv"
"$PY" "$SCRIPT_DIR/merge_eval40.py" --expected-conditions 40 \
  --prefix "$OLD_EVAL/generated/denovo_6p7p_p19.raw_k8.csv" --extension "$OUT/generated/denovo_6p7p.ranks_9_40.csv" \
  --raw-output "$OUT/merged/denovo_6p7p.raw40.csv" --eval-output "$OUT/merged/denovo_6p7p.eval40.csv"
"$PY" "$SCRIPT_DIR/merge_eval40.py" --expected-conditions 12 \
  --prefix "$OUT/generated/ood.raw_k8.csv" --extension "$OUT/generated/ood.ranks_9_40.csv" \
  --raw-output "$OUT/merged/ood.raw40.csv" --eval-output "$OUT/merged/ood.eval40.csv"

"$PY" "$REPO/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
  --eval-csv "$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020/data/denovo_2p4p.reference.csv" \
  --candidate-csv "$OUT/merged/denovo_2p4p.eval40.csv" --output-dir "$OUT/results/denovo_2p4p" \
  --budgets 1,4,8,20,40 --report-title "P23 aligned 24k de novo 2p-4p paper protocol" \
  --candidate-description "ordered frozen direct-LLM candidates"
"$PY" "$REPO/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
  --eval-csv "$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data/denovo_expanded.reference.csv" \
  --candidate-csv "$OUT/merged/denovo_6p7p.eval40.csv" --output-dir "$OUT/results/denovo_6p7p" \
  --budgets 1,4,8,20,40 --report-title "P23 aligned 24k de novo 6p-7p paper diagnostic" \
  --candidate-description "ordered frozen direct-LLM candidates"
"$PY" "$SCRIPT_DIR/evaluate_ood_12.py" --reference "$OUT/data/sketchmol_ood_12.reference.csv" \
  --candidates "$OUT/merged/ood.eval40.csv" --output-dir "$OUT/results/ood_12"
touch "$OUT/PAPER_EVAL_COMPLETE"
echo "P23 paper-protocol scoring complete: $OUT"
