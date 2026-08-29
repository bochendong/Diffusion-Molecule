#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P311_SCRIPT_DIR:?P311_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P311_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
OUT="${P311_OUTPUT_ROOT:-$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101}"
"$PY" "$PROJECT/experiments/p30_1_alignment_raw1_rl/build_small_raw1_gate.py" \
  --prompts "$P23/eval_corrected_prompts/prompts/denovo_2p4p_p20.jsonl" \
  --prompts "$P23/eval_denovo_table1_fill/data/denovo_5p.prompts.jsonl" \
  --prompts "$P23/eval_corrected_prompts/prompts/denovo_6p7p_p19.jsonl" \
  --baseline-detail "$P24/eval_table1/results/denovo_2p4p/budget_sweep_condition_detail.csv" \
  --baseline-detail "$P24/eval_table1/results/denovo_5p/budget_sweep_condition_detail.csv" \
  --baseline-detail "$P24/eval_table1/results/denovo_6p7p/budget_sweep_condition_detail.csv" \
  --output-dir "$OUT/gate/data" --per-arity 100 --seed 31131
touch "$OUT/GATE_DATA_COMPLETE"
