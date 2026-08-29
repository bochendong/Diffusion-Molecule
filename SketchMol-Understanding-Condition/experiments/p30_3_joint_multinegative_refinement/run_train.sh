#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P303_SCRIPT_DIR:?P303_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P303_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P303_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P303_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P301="$PROJECT/outputs/p30_1_alignment_raw1_rl/seed_30101"
INPUT_ADAPTER="${P303_INPUT_ADAPTER:-$P301/model/balanced_shared_rl/checkpoint-030/adapter}"
OUT="${P303_OUTPUT_ROOT:-$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301}"
for path in "$OUT/PREPARED" "$OUT/data/train.invalid_balanced.jsonl" "$INPUT_ADAPTER/adapter_model.safetensors"; do
  test -f "$path"
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$PROJECT/experiments/p23_explicit_task_stage1_v2/train_contrastive.py" \
  --train-jsonl "$OUT/data/train.invalid_balanced.jsonl" \
  --output-dir "$OUT/model/refined" --base-model "$BASE" \
  --input-adapter "$INPUT_ADAPTER" --epochs 0.5 --learning-rate 2e-6 \
  --gradient-accumulation 16 --seed 30301
sha256sum "$OUT/model/refined/adapter/adapter_model.safetensors" > "$OUT/model/refined/FROZEN.sha256"
touch "$OUT/TRAIN_COMPLETE"
