#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P33_SCRIPT_DIR:?P33_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P33_OUTPUT_ROOT:-$PROJECT/outputs/p33_joint_vs_separate_pilot/seed_33001}"
PY="${P33_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
"$PY" "$SCRIPT_DIR/prepare_pilot.py" \
  --train-source "$P24/alignment_refresh/data/train.sft.jsonl" \
  --denovo-prompts "$P23/eval_corrected_prompts/prompts/denovo_2p4p_p20.jsonl" \
  --denovo-prompts "$P23/eval_denovo_table1_fill/data/denovo_5p.prompts.jsonl" \
  --denovo-prompts "$P23/eval_corrected_prompts/prompts/denovo_6p7p_p19.jsonl" \
  --output-dir "$OUT/data" --seed 33001
