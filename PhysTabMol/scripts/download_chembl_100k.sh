#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PHYSTABMOL_SUPPRESS_RDKIT_LOGS="${PHYSTABMOL_SUPPRESS_RDKIT_LOGS:-1}"
export PHYSTABMOL_PROGRESS="${PHYSTABMOL_PROGRESS:-1}"
export PHYSTABMOL_PROGRESS_STEP="${PHYSTABMOL_PROGRESS_STEP:-5}"

python3 scripts/download_dataset.py \
  --source chembl \
  --limit "${PHYSTABMOL_LIMIT:-100000}" \
  --out "${PHYSTABMOL_OUT:-data/molecules.csv}" \
  "$@"
