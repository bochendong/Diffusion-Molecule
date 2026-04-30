#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/download_dataset.py \
  --source chembl \
  --limit "${PHYSTABMOL_LIMIT:-100000}" \
  --out "${PHYSTABMOL_OUT:-data/molecules.csv}" \
  "$@"
