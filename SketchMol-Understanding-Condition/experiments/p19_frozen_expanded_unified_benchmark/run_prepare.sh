#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P19_SCRIPT_DIR:?P19_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P19_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P19_OUTPUT_ROOT:-$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919}"
mkdir -p "$OUT"
exec 9>"$OUT/.prepare.lock"
if ! flock -n 9; then
  for _ in $(seq 1 150); do
    if [[ -f "$OUT/PREPARED_BEFORE_GENERATION" ]]; then
      echo "P19 prepare already completed by racer"
      exit 0
    fi
    sleep 2
  done
  echo "timed out waiting for P19 prepare lock" >&2
  exit 1
fi
if [[ -f "$OUT/PREPARED_BEFORE_GENERATION" ]]; then
  echo "P19 already prepared"
  exit 0
fi
P16="$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P18="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12"
P6="$PROJECT/outputs/p6_unified_transition_policy_v1/seed_7/data"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
export PYTHONPATH="$P17_SCRIPT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/data"
test -f "$SCRIPT_DIR/preregistration.json"
"$PY" "$SCRIPT_DIR/build_expanded_subsets.py" \
  --table1-csv "$P6/edit_table1_gate.csv" \
  --denovo-csv "$P6/denovo_hard_gate.csv" \
  --p16-train-jsonl "$P16/data/train.mixed.jsonl" \
  --p17-train-jsonl "$P17/data/train.paired.jsonl" \
  --p17-table-reference "$P17/pilot/table1_pilot.reference.csv" \
  --p17-denovo-reference "$P17/pilot/denovo_pilot.reference.csv" \
  --output-dir "$OUT/data" --seed 1717
(cd "$OUT/data" && sha256sum -c LOCKED.sha256)
sha256sum \
  "$P17/model/p17/adapter/adapter_model.safetensors" \
  "$P18/model/p18/adapter/adapter_model.safetensors" > "$OUT/data/ADAPTERS_LOCKED.sha256"
sha256sum -c "$OUT/data/ADAPTERS_LOCKED.sha256"
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
touch "$OUT/PREPARED_BEFORE_GENERATION"
echo "P19 subsets frozen before generation: $OUT"
