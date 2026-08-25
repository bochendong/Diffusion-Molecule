#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P19_SCRIPT_DIR:?P19_SCRIPT_DIR must be exported by submitter}"
MODEL="${P19_MODEL:?P19_MODEL must be p17 or p18}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P19_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P19_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P19_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P19_OUTPUT_ROOT:-$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919}"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
mkdir -p "$OUT"
exec 9>"$OUT/.${MODEL}.generation.lock"
if ! flock -n 9; then
  echo "another $MODEL generation racer owns the output lock" >&2
  exit 75
fi
if [[ -f "$OUT/${MODEL^^}_GENERATION_COMPLETE" ]]; then
  echo "P19 $MODEL generation already complete"
  exit 0
fi
case "$MODEL" in
  p17)
    ADAPTER="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717/model/p17/adapter"
    LABEL="p17_frozen_expanded"
    ;;
  p18)
    ADAPTER="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12/model/p18/adapter"
    LABEL="p18_frozen_expanded"
    ;;
  *) echo "unsupported P19_MODEL=$MODEL" >&2; exit 2 ;;
esac
test -f "$OUT/PREPARED_BEFORE_GENERATION"
(cd "$OUT/data" && sha256sum -c LOCKED.sha256)
sha256sum -c "$OUT/data/ADAPTERS_LOCKED.sha256"
test -d "$ADAPTER"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$P17_SCRIPT:$SCRIPT_DIR:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/generated/$MODEL"
"$PY" "$P17_SCRIPT/generate_pilot.py" \
  --prompts-jsonl "$OUT/data/table1_expanded.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/$MODEL/table1.raw.csv" --seed 2717
"$PY" "$SCRIPT_DIR/relabel_generated.py" --csv "$OUT/generated/$MODEL/table1.raw.csv" --label "$LABEL"
"$PY" "$P17_SCRIPT/generate_pilot.py" \
  --prompts-jsonl "$OUT/data/denovo_expanded.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/$MODEL/denovo.raw.csv" --seed 3717
"$PY" "$SCRIPT_DIR/relabel_generated.py" --csv "$OUT/generated/$MODEL/denovo.raw.csv" --label "$LABEL"
"$PY" "$SCRIPT_DIR/validate_generated.py" --generated-dir "$OUT/generated/$MODEL" --model "$MODEL"
touch "$OUT/${MODEL^^}_GENERATION_COMPLETE"
echo "P19 $MODEL frozen raw generation complete"
