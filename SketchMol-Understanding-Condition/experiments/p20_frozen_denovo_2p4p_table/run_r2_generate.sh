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
test -f "$OUT/r2/PREPARED_BEFORE_EXTENSION"
sha256sum -c "$OUT/r2/ADAPTERS_LOCKED.sha256"
mkdir -p "$OUT/r2/generated/$MODEL"
exec 9>"$OUT/r2/.${MODEL}.extension.lock"
flock -n 9 || exit 75
test -f "$OUT/r2/${MODEL^^}_EXTENSION_COMPLETE" && exit 0
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$P17_SCRIPT:$HERE:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$HERE/generate_extension.py" \
  --prompts-jsonl "$OUT/data/denovo_2p4p.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --output-csv "$OUT/r2/generated/$MODEL/ranks_9_40.csv" --model-label "$MODEL" --seed 9040
"$PY" - <<PY
import csv,collections
p='$OUT/r2/generated/$MODEL/ranks_9_40.csv'
with open(p,newline='') as f:r=list(csv.DictReader(f))
g=collections.defaultdict(list)
for x in r:g[x['condition_id']].append(int(float(x['candidate_rank'])))
assert len(r)==9600 and len(g)==300 and all(sorted(v)==list(range(9,41)) for v in g.values())
print('validated extension',len(r),'rows',len(g),'conditions')
PY
touch "$OUT/r2/${MODEL^^}_EXTENSION_COMPLETE"
if [[ "$MODEL" == p17 && "${P20_CHAIN_GUARD:-0}" != 1 && ! -f "$OUT/r2/P18_EXTENSION_COMPLETE" ]]; then
  gpu_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$gpu_mib" =~ ^[0-9]+$ && "$gpu_mib" -gt 30000 ]]; then
    echo "P20-R2 chaining frozen P18 extension in the same >30GB allocation"
    set +e
    P20_MODEL=p18 P20_CHAIN_GUARD=1 bash "$HERE/run_r2_generate.sh"
    chained_status=$?
    set -e
    if [[ "$chained_status" -ne 0 && "$chained_status" -ne 75 ]]; then
      exit "$chained_status"
    fi
  fi
fi
