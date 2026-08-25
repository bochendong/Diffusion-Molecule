#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P23_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P23_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323}"
P16="${P23_INPUT_ADAPTER:-$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616/model/mixed/adapter}"

[[ -f "$OUT/PREPARED" ]] || { echo "ERROR: P23 is not prepared" >&2; exit 2; }
[[ -f "$P16/adapter_model.safetensors" ]] || { echo "ERROR: missing P16 adapter: $P16" >&2; exit 2; }
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_sft.py" \
  --train-jsonl "$OUT/data/train.sft.jsonl" --output-dir "$OUT/model/sft" \
  --base-model "$BASE" --input-adapter "$P16" \
  --epochs "${P23_SFT_EPOCHS:-0.5}" --learning-rate "${P23_SFT_LR:-2e-5}" \
  --gradient-accumulation "${P23_GRADIENT_ACCUMULATION:-16}" --seed 2323
sha256sum "$OUT/model/sft/adapter/adapter_model.safetensors" > "$OUT/model/sft/FROZEN.sha256"
touch "$OUT/SFT_COMPLETE"
echo "P23 positive SFT complete: $OUT"
