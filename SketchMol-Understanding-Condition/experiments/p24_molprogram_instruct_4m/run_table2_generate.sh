#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
P23_OUT="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P23="$PROJECT/experiments/p23_explicit_task_stage1_v2"
P19="$PROJECT/experiments/p19_frozen_expanded_unified_benchmark"
BASE="${P24_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P24_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
ADAPTER="$P24_OUT/alignment_refresh/model/adapter"
FROZEN="$P23_OUT/eval_moledit_table1_500/data"
OUT="$P24_OUT/eval_table2"
test -f "$P24_OUT/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE"
for path in "$ADAPTER/adapter_model.safetensors" "$FROZEN/table1_500.prompts.jsonl" \
  "$FROZEN/table1_500.reference.csv" "$FROZEN/table1_500.manifest.json"; do
  test -f "$path"
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$P23${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/generated"
"$PY" "$P23/generate_sampled_once.py" --prompts-jsonl "$FROZEN/table1_500.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/table2_500.sampled_once.csv" --seed 23501 --batch-size 8
"$PY" "$P19/relabel_generated.py" --csv "$OUT/generated/table2_500.sampled_once.csv" \
  --label p24_balanced_refresh_sampled_once
touch "$OUT/TABLE2_GENERATION_COMPLETE"
echo "P24 Table 2 generation complete: $OUT"
