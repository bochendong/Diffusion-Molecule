#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P37_SCRIPT_DIR:?P37_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P37_OUTPUT_ROOT:-$PROJECT/outputs/p37_denovo_overlap_expanded_eval/seed_37101}"
PY="${P37_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11
"$PY" "$SCRIPT_DIR/collect_overlap.py" --output-root "$OUT" \
  --bootstrap-seed 37201 --bootstrap-replicates 100000
