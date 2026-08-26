#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P23_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P23_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
OUT="${P23_EDIT500_OUTPUT_ROOT:-$TRAIN_OUT/eval_moledit_table1_500}"
P17="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
test -f "$OUT/EDIT500_PREPARED"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$P17${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/generate_sampled_once.py" \
  --prompts-jsonl "$OUT/data/table1_500.prompts.jsonl" --base-model "$BASE" \
  --adapter-dir "$TRAIN_OUT/model/stage1_v2/adapter" \
  --output-csv "$OUT/generated/table1_500.sampled_once.csv" --seed 23501 --batch-size 8
touch "$OUT/EDIT500_GENERATED"
