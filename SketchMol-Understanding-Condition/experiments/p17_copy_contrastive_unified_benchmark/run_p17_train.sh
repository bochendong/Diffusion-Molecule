#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P17_SCRIPT_DIR:?P17_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"; PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P17_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P17_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P17_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P17_OUTPUT_ROOT:-$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717}"
P16="$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616/model/mixed/adapter"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6; fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}" HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_copy_contrastive.py" --train-jsonl "$OUT/data/train.paired.jsonl" --output-dir "$OUT/model/p17" \
  --base-model "$BASE" --input-adapter "$P16" --epochs "${P17_EPOCHS:-1.0}" --learning-rate 2.5e-5 \
  --pairwise-margin 0.20 --pairwise-weight 0.35 --gradient-accumulation 8 --seed 1717
sha256sum "$OUT/model/p17/adapter/adapter_model.safetensors" > "$OUT/model/p17/FROZEN.sha256"
touch "$OUT/P17_TRAIN_COMPLETE"
