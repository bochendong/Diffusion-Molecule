#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P18_SCRIPT_DIR:?P18_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"; PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P18_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P18_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P18_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P18_OUTPUT_ROOT:-$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818}"
P16="$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616/model/mixed/adapter"
test -f "$OUT/PREPARED"; test -f "$P16/adapter_model.safetensors"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6; fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}" HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_multinegative.py" --train-jsonl "$OUT/data/train.multinegative.jsonl" --output-dir "$OUT/model/p18" \
  --base-model "$BASE" --input-adapter "$P16" --epochs 1.0 --learning-rate 2e-5 --gradient-accumulation 8 --seed 1818
sha256sum "$OUT/model/p18/adapter/adapter_model.safetensors" > "$OUT/model/p18/FROZEN.sha256"
touch "$OUT/P18_TRAIN_COMPLETE"
