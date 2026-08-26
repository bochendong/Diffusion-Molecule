#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P23_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P23_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P23_DENOVO_FILL_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/eval_denovo_table1_fill}"
ALIGNED="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/model/stage1_v2/adapter"
LEGACY="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12/model/p18/adapter"
P19="$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919"
P19_SCRIPT="$PROJECT/experiments/p19_frozen_expanded_unified_benchmark"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P20_SCRIPT="$PROJECT/experiments/p20_frozen_denovo_2p4p_table"
test -f "$OUT/DENOVO_FILL_PREPARED"
test -f "$P19/generated/p18/denovo.raw.csv"
for path in "$BASE" "$ALIGNED" "$LEGACY"; do
  [[ -e "$path" ]] || { echo "ERROR: missing generation input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$P17_SCRIPT:$P20_SCRIPT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/generated/aligned" "$OUT/generated/legacy"

"$PY" "$P17_SCRIPT/generate_pilot.py" --prompts-jsonl "$OUT/data/denovo_5p.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$ALIGNED" --output-csv "$OUT/generated/aligned/denovo_5p.raw8.csv" --seed 23233
"$PY" "$P19_SCRIPT/relabel_generated.py" --csv "$OUT/generated/aligned/denovo_5p.raw8.csv" --label p23_aligned24k
"$PY" "$P20_SCRIPT/generate_extension.py" --prompts-jsonl "$OUT/data/denovo_5p.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$ALIGNED" --output-csv "$OUT/generated/aligned/denovo_5p.ranks_9_40.csv" \
  --model-label p23_aligned24k --seed 232340
"$PY" "$P17_SCRIPT/generate_pilot.py" --prompts-jsonl "$OUT/data/denovo_5p.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$LEGACY" --output-csv "$OUT/generated/legacy/denovo_5p.raw8.csv" --seed 4020
"$PY" "$P19_SCRIPT/relabel_generated.py" --csv "$OUT/generated/legacy/denovo_5p.raw8.csv" --label p18_legacy160
"$PY" "$P20_SCRIPT/generate_extension.py" --prompts-jsonl "$OUT/data/denovo_5p.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$LEGACY" --output-csv "$OUT/generated/legacy/denovo_5p.ranks_9_40.csv" \
  --model-label p18_legacy160 --seed 9040
"$PY" "$P20_SCRIPT/generate_extension.py" --prompts-jsonl "$P19/data/denovo_expanded.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$LEGACY" --output-csv "$OUT/generated/legacy/denovo_6p7p.ranks_9_40.csv" \
  --model-label p18_legacy160 --seed 9040
touch "$OUT/DENOVO_FILL_GENERATED"
