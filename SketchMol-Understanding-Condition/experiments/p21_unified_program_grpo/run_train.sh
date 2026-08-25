#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P21_SCRIPT_DIR:?P21_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P21_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P21_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P21_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P21_OUTPUT_ROOT:-$PROJECT/outputs/p21_unified_program_grpo/seed_2121}"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P18="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12"
test -f "$OUT/PREPARED"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6; fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p17_copy_contrastive_unified_benchmark:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_unified_grpo.py" \
  --train-jsonl "$P17/data/train.paired.jsonl" --base-model "$BASE" \
  --input-adapter "$P18/model/p18/adapter" --output-dir "$OUT/model/p21" \
  --max-prompts 128 --group-size 4 --learning-rate 1e-6 --sft-anchor-weight 0.20 --seed 2121
sha256sum "$OUT/model/p21/adapter/adapter_model.safetensors" > "$OUT/model/p21/FROZEN.sha256"
touch "$OUT/TRAIN_COMPLETE"
echo "P21 GRPO checkpoint frozen"
