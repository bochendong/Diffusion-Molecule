#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${P16_SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$P16_SCRIPT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/p16_protocol.py" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${P16_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${P16_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE_MODEL="${P16_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P16_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p16_direct_llm_unified_generation_editing/seed_1616}"
DENOVO_CSV="${P16_DENOVO_CSV:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_train_rows.csv}"
EDIT_CSV="${P16_EDIT_CSV:-$PROJECT_DIR/outputs/p8_1_1_short_transaction_r1/seed_7/data/edit_train.csv}"
TRAIN_PER_MODE="${P16_TRAIN_PER_MODE:-128}"
DEV_PER_MODE="${P16_DEV_PER_MODE:-8}"
EPOCHS="${P16_EPOCHS:-1.0}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
for path in "$DENOVO_CSV" "$EDIT_CSV"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P16 input: $path" >&2; exit 2; }
done
[[ -d "$BASE_MODEL" ]] || { echo "ERROR: cached Qwen checkpoint missing: $BASE_MODEL" >&2; exit 2; }
[[ -d "$DEP_OVERLAY/peft" ]] || { echo "ERROR: PEFT overlay missing: $DEP_OVERLAY" >&2; exit 2; }
export PYTHONPATH="$DEP_OVERLAY:$SCRIPT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/data" "$OUT/model" "$OUT/eval"

"$PYTHON_BIN" "$SCRIPT_DIR/build_direct_sft.py" \
  --denovo-csv "$DENOVO_CSV" --edit-csv "$EDIT_CSV" --output-dir "$OUT/data" \
  --train-per-mode "$TRAIN_PER_MODE" --dev-per-mode "$DEV_PER_MODE" --seed 1616

for arm in mixed denovo edit; do
  "$PYTHON_BIN" "$SCRIPT_DIR/train_lora.py" \
    --train-jsonl "$OUT/data/train.$arm.jsonl" --output-dir "$OUT/model/$arm" \
    --base-model "$BASE_MODEL" --arm "$arm" --epochs "$EPOCHS" --seed 1616
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_direct.py" \
    --dev-jsonl "$OUT/data/dev.jsonl" --base-model "$BASE_MODEL" \
    --adapter-dir "$OUT/model/$arm/adapter" --arm "$arm" --output-json "$OUT/eval/$arm.json" \
    --limit-per-mode "${P16_EVAL_PER_MODE:-8}" --fixed-k "${P16_FIXED_K:-3}" --seed 1616
done

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_results.py" \
  --mixed "$OUT/eval/mixed.json" --denovo "$OUT/eval/denovo.json" --edit "$OUT/eval/edit.json" \
  --manifest "$OUT/data/manifest.json" --output "$OUT/summary.json"
touch "$OUT/COMPLETE"
echo "P16 pilot complete: $OUT"
