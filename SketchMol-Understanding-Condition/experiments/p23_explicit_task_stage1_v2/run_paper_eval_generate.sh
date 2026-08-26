#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P23_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P23_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
OUT="${P23_EVAL_OUTPUT_ROOT:-$TRAIN_OUT/eval_paper_protocol}"
OLD_EVAL="$TRAIN_OUT/eval_corrected_prompts"
ADAPTER="$TRAIN_OUT/model/stage1_v2/adapter"
P17="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P20="$PROJECT/experiments/p20_frozen_denovo_2p4p_table"

test -f "$TRAIN_OUT/STAGE1_V2_COMPLETE"
test -f "$OLD_EVAL/GENERATION_COMPLETE"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$P17${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/data" "$OUT/generated"

"$PY" "$SCRIPT_DIR/prepare_paper_eval.py" --output-dir "$OUT/data"
"$PY" "$P20/generate_extension.py" \
  --prompts-jsonl "$OLD_EVAL/prompts/denovo_2p4p_p20.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/denovo_2p4p.ranks_9_40.csv" --model-label p23_aligned24k --seed 232340
"$PY" "$P20/generate_extension.py" \
  --prompts-jsonl "$OLD_EVAL/prompts/denovo_6p7p_p19.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/denovo_6p7p.ranks_9_40.csv" --model-label p23_aligned24k --seed 232341
"$PY" "$P17/generate_pilot.py" \
  --prompts-jsonl "$OUT/data/sketchmol_ood_12.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/ood.raw_k8.csv" --seed 232342
"$PY" "$P20/generate_extension.py" \
  --prompts-jsonl "$OUT/data/sketchmol_ood_12.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/ood.ranks_9_40.csv" --model-label p23_aligned24k --seed 232343
touch "$OUT/PAPER_GENERATION_COMPLETE"
echo "P23 paper-protocol generation complete: $OUT"
