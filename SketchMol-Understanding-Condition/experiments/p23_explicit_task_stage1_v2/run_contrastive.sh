#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P23_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P23_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323}"
SFT="$OUT/model/sft/adapter"

[[ -f "$OUT/SFT_COMPLETE" ]] || { echo "ERROR: P23 SFT is incomplete" >&2; exit 2; }
[[ -f "$SFT/adapter_model.safetensors" ]] || { echo "ERROR: missing P23 SFT adapter" >&2; exit 2; }
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_contrastive.py" \
  --train-jsonl "$OUT/data/train.contrastive.jsonl" --output-dir "$OUT/model/stage1_v2" \
  --base-model "$BASE" --input-adapter "$SFT" \
  --epochs "${P23_CONTRASTIVE_EPOCHS:-0.25}" --learning-rate "${P23_CONTRASTIVE_LR:-1e-5}" \
  --gradient-accumulation "${P23_GRADIENT_ACCUMULATION:-16}" --seed 2323
sha256sum "$OUT/model/stage1_v2/adapter/adapter_model.safetensors" > "$OUT/model/stage1_v2/FROZEN.sha256"
touch "$OUT/STAGE1_V2_COMPLETE"
echo "P23 Stage-1 v2 complete: $OUT"
