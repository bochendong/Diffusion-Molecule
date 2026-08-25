#!/usr/bin/env bash
set -euo pipefail
HERE="${P20_SCRIPT_DIR:?P20_SCRIPT_DIR must be exported}"
MODEL="${P20_MODEL:?P20_MODEL must be p17 or p18}"
PROJECT="$(cd "$HERE/../.." && pwd)"
PY="${P20_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P20_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P20_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P20_OUTPUT_ROOT:-$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020}"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
case "$MODEL" in
  p17) ADAPTER="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717/model/p17/adapter" ;;
  p18) ADAPTER="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12/model/p18/adapter" ;;
  *) exit 2 ;;
esac
test -f "$OUT/PREPARED_BEFORE_GENERATION"
(cd "$OUT/data" && sha256sum -c LOCKED.sha256)
sha256sum -c "$OUT/data/ADAPTERS_LOCKED.sha256"
mkdir -p "$OUT/generated/$MODEL"
exec 9>"$OUT/.${MODEL}.generation.lock"
flock -n 9 || exit 75
test -f "$OUT/${MODEL^^}_GENERATION_COMPLETE" && exit 0
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$P17_SCRIPT:$HERE:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$P17_SCRIPT/generate_pilot.py" \
  --prompts-jsonl "$OUT/data/denovo_2p4p.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/generated/$MODEL/denovo.raw.csv" --seed 4020
"$PY" - <<PY
import csv
p='$OUT/generated/$MODEL/denovo.raw.csv'
with open(p,newline='') as f:r=list(csv.DictReader(f))
assert len(r)==2400 and len({x['condition_id'] for x in r})==300
assert all(sum(y['condition_id']==x for y in r)==8 for x in {z['condition_id'] for z in r})
print('validated',len(r),'records')
PY
touch "$OUT/${MODEL^^}_GENERATION_COMPLETE"
if [[ "$MODEL" == p17 && "${P20_CHAIN_GUARD:-0}" != 1 ]]; then
  gpu_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$gpu_mib" =~ ^[0-9]+$ && "$gpu_mib" -gt 30000 && ! -f "$OUT/P18_GENERATION_COMPLETE" ]]; then
    echo "P20 chaining frozen P18 generation inside the same >30GB allocation"
    set +e
    P20_MODEL=p18 P20_CHAIN_GUARD=1 bash "$HERE/run_generate.sh"
    chained_status=$?
    set -e
    if [[ "$chained_status" -ne 0 && "$chained_status" -ne 75 ]]; then
      exit "$chained_status"
    fi
  fi
fi
