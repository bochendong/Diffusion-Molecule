#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
P23_OUT="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
BASE="${P24_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P24_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P17="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P19="$PROJECT/experiments/p19_frozen_expanded_unified_benchmark"
P20="$PROJECT/experiments/p20_frozen_denovo_2p4p_table"
OUT="$P24_OUT/eval_table1"
ADAPTER="$P24_OUT/alignment_refresh/model/adapter"
PROMPTS_2P4P="$P23_OUT/eval_corrected_prompts/prompts/denovo_2p4p_p20.jsonl"
PROMPTS_5P="$P23_OUT/eval_denovo_table1_fill/data/denovo_5p.prompts.jsonl"
PROMPTS_6P7P="$P23_OUT/eval_corrected_prompts/prompts/denovo_6p7p_p19.jsonl"
test -f "$P24_OUT/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE"
for path in "$ADAPTER/adapter_model.safetensors" "$PROMPTS_2P4P" "$PROMPTS_5P" "$PROMPTS_6P7P"; do
  test -f "$path"
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$P17:$P20${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/generated"

generate_pool() {
  local name="$1" prompts="$2" prefix_seed="$3" extension_seed="$4"
  "$PY" "$P17/generate_pilot.py" --prompts-jsonl "$prompts" --base-model "$BASE" \
    --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/${name}.raw8.csv" --seed "$prefix_seed"
  "$PY" "$P19/relabel_generated.py" --csv "$OUT/generated/${name}.raw8.csv" --label p24_balanced_refresh
  "$PY" "$P20/generate_extension.py" --prompts-jsonl "$prompts" --base-model "$BASE" \
    --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/${name}.ranks_9_40.csv" \
    --model-label p24_balanced_refresh --seed "$extension_seed"
}

generate_pool denovo_2p4p "$PROMPTS_2P4P" 24031 240310
generate_pool denovo_5p "$PROMPTS_5P" 24032 240320
generate_pool denovo_6p7p "$PROMPTS_6P7P" 24033 240330
touch "$OUT/TABLE1_GENERATION_COMPLETE"
echo "P24 Table 1 generation complete: $OUT"
