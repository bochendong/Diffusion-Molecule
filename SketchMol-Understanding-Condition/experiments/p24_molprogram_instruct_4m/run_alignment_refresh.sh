#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P23_OUT="${P24_P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
BASE="${P24_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P24_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

input="$P23_OUT/data/train.sft.jsonl"
refresh="$P24_OUT/alignment_refresh/data/train.sft.jsonl"
test -f "$P24_OUT/full/TRAINING_COMPLETE"
"$PY" "$SCRIPT_DIR/build_alignment_refresh.py" \
  --input-jsonl "$input" --output-jsonl "$refresh" \
  --summary-json "$P24_OUT/alignment_refresh/data/summary.json" --rows-per-task 720
"$PY" "$PROJECT/experiments/p23_explicit_task_stage1_v2/train_sft.py" \
  --train-jsonl "$refresh" --output-dir "$P24_OUT/alignment_refresh/model" \
  --base-model "$BASE" --input-adapter "$P24_OUT/full/adapter" \
  --epochs 1.0 --learning-rate 5e-6 --gradient-accumulation 16 --seed 24004
sha256sum "$P24_OUT/alignment_refresh/model/adapter/adapter_model.safetensors" \
  > "$P24_OUT/alignment_refresh/model/FROZEN.sha256"
touch "$P24_OUT/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE"
echo "P24 alignment refresh complete: $P24_OUT/alignment_refresh"
