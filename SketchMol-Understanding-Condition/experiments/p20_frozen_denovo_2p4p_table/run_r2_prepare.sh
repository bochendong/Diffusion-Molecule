#!/usr/bin/env bash
set -euo pipefail
HERE="${P20_SCRIPT_DIR:?P20_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$HERE/../.." && pwd)"
PY="${P20_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P20_OUTPUT_ROOT:-$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020}"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P18="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12"
test -f "$OUT/FINAL_COMPLETE"
mkdir -p "$OUT/r2"
exec 9>"$OUT/r2/.prepare.lock"
flock -n 9 || exit 75
test -f "$OUT/r2/PREPARED_BEFORE_EXTENSION" && exit 0
"$PY" "$HERE/prepare_r2.py" \
  --preregistration "$HERE/r2_preregistration.json" --r1-manifest "$OUT/data/manifest.json" \
  --reference "$OUT/data/denovo_2p4p.reference.csv" --prompts "$OUT/data/denovo_2p4p.prompts.jsonl" \
  --p17-prefix "$OUT/generated/p17/denovo.raw.csv" --p18-prefix "$OUT/generated/p18/denovo.raw.csv" \
  --output "$OUT/r2/audit_before_extension.json"
sha256sum "$P17/model/p17/adapter/adapter_model.safetensors" "$P18/model/p18/adapter/adapter_model.safetensors" > "$OUT/r2/ADAPTERS_LOCKED.sha256"
sha256sum -c "$OUT/r2/ADAPTERS_LOCKED.sha256"
touch "$OUT/r2/PREPARED_BEFORE_EXTENSION"
