#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"$PYTHON_BIN" -m molpilot.build_repair_dataset \
  --data "${MOLPILOT_DATA:-../PhysTabMol/data/molecules.csv}" \
  --output-dir "${MOLPILOT_REPAIR_DATASET_DIR:-outputs/repair_dataset}" \
  --limit "${MOLPILOT_LIMIT:-1000}" \
  --repair-corruptions "${MOLPILOT_REPAIR_CORRUPTIONS:-}" \
  --repair-corruptions-per-molecule "${MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE:-2}" \
  --seed "${MOLPILOT_SEED:-7}"
