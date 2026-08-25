#!/usr/bin/env bash
set -euo pipefail
HERE="${P20_SCRIPT_DIR:?P20_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$HERE/../.." && pwd)"
PY="${P20_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P20_OUTPUT_ROOT:-$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020}"
test -f "$OUT/P17_GENERATION_COMPLETE"
test -f "$OUT/P18_GENERATION_COMPLETE"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$PROJECT:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$HERE/aggregate.py" \
  --manifest "$OUT/data/manifest.json" --reference "$OUT/data/denovo_2p4p.reference.csv" \
  --p17-candidates "$OUT/generated/p17/denovo.raw.csv" --p18-candidates "$OUT/generated/p18/denovo.raw.csv" \
  --output-dir "$OUT/results"
touch "$OUT/FINAL_COMPLETE"
