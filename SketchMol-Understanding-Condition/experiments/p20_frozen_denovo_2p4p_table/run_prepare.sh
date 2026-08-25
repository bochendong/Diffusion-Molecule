#!/usr/bin/env bash
set -euo pipefail
HERE="${P20_SCRIPT_DIR:?P20_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$HERE/../.." && pwd)"
PY="${P20_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P20_OUTPUT_ROOT:-$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020}"
DIRECT="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
P16="$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P18="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12"
mkdir -p "$OUT/data"
exec 9>"$OUT/.prepare.lock"
if ! flock -n 9; then
  for _ in $(seq 1 150); do test -f "$OUT/PREPARED_BEFORE_GENERATION" && exit 0; sleep 2; done
  exit 1
fi
if test -f "$OUT/PREPARED_BEFORE_GENERATION"; then exit 0; fi
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$HERE/build_subsets.py" \
  --eval-csv "$DIRECT/denovo_2p7p_eval_rows.csv" \
  --p16-train-jsonl "$P16/data/train.mixed.jsonl" \
  --p17-train-jsonl "$P17/data/train.paired.jsonl" \
  --output-dir "$OUT/data" --seed 2020 --per-count 100
(cd "$OUT/data" && sha256sum -c LOCKED.sha256)
sha256sum "$P17/model/p17/adapter/adapter_model.safetensors" "$P18/model/p18/adapter/adapter_model.safetensors" > "$OUT/data/ADAPTERS_LOCKED.sha256"
sha256sum -c "$OUT/data/ADAPTERS_LOCKED.sha256"
touch "$OUT/PREPARED_BEFORE_GENERATION"
