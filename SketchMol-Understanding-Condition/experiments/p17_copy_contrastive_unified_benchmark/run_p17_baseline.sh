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
export PYTHONPATH="$DEP:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for view in id ood; do
  file="$OUT/data/dev.$([[ "$view" == id ]] && echo id_condition_source_isolated || echo condition_source_ood).jsonl"
  "$PY" "$SCRIPT_DIR/evaluate_expanded.py" --dev-jsonl "$file" --base-model "$BASE" --adapter-dir "$P16" \
    --label p16_frozen --view "$view" --output-json "$OUT/eval/p16.$view.json" --fixed-k 3 --seed 1717
done
touch "$OUT/P16_BASELINE_COMPLETE"
