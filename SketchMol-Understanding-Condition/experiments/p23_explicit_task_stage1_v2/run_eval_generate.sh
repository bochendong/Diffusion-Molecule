#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P23_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P23_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_fast}"
OUT="${P23_EVAL_OUTPUT_ROOT:-$TRAIN_OUT/eval_corrected_prompts}"
ADAPTER="$TRAIN_OUT/model/stage1_v2/adapter"
P17="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P19_DATA="$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data"
P20_DATA="$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020/data"

[[ -f "$TRAIN_OUT/STAGE1_V2_COMPLETE" ]] || { echo "ERROR: P23 Stage 1 is incomplete" >&2; exit 2; }
for path in "$ADAPTER/adapter_model.safetensors" "$P19_DATA/table1_expanded.reference.csv" \
  "$P19_DATA/denovo_expanded.reference.csv" "$P20_DATA/denovo_2p4p.reference.csv"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P23 evaluation input: $path" >&2; exit 2; }
done
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$P17${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/prompts" "$OUT/generated"

"$PY" "$SCRIPT_DIR/rebuild_eval_prompts.py" --mode edit \
  --reference "$P19_DATA/table1_expanded.reference.csv" --output "$OUT/prompts/edit_p19.jsonl"
"$PY" "$SCRIPT_DIR/rebuild_eval_prompts.py" --mode de_novo \
  --reference "$P19_DATA/denovo_expanded.reference.csv" --output "$OUT/prompts/denovo_6p7p_p19.jsonl"
"$PY" "$SCRIPT_DIR/rebuild_eval_prompts.py" --mode de_novo \
  --reference "$P20_DATA/denovo_2p4p.reference.csv" --output "$OUT/prompts/denovo_2p4p_p20.jsonl"

"$PY" "$P17/generate_pilot.py" --prompts-jsonl "$OUT/prompts/edit_p19.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/edit_p19.raw_k8.csv" --seed 23231
"$PY" "$P17/generate_pilot.py" --prompts-jsonl "$OUT/prompts/denovo_6p7p_p19.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/denovo_6p7p_p19.raw_k8.csv" --seed 23232
"$PY" "$P17/generate_pilot.py" --prompts-jsonl "$OUT/prompts/denovo_2p4p_p20.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/denovo_2p4p_p20.raw_k8.csv" --seed 23233
touch "$OUT/GENERATION_COMPLETE"
echo "P23 corrected-prompt generation complete: $OUT"
