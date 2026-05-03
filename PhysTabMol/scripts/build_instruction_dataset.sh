#!/bin/bash
set -euo pipefail

# Build the no-human-label verified instruction-editing dataset.
#
# Direct server usage:
#   cd PhysTabMol
#   bash scripts/build_instruction_dataset.sh
#
# Useful overrides:
#   PHYSTABMOL_DATA=data/molecules.csv PHYSTABMOL_MAX_PAIRS=100000 bash scripts/build_instruction_dataset.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
# shellcheck source=/dev/null
source "$PHYSTABMOL_ROOT/scripts/ensure_phystabmol_venv.sh"

DATA="${PHYSTABMOL_DATA:-data/molecules.csv}"
OUT="${PHYSTABMOL_INSTRUCTION_OUT:-data/instruction_editing.csv}"
JSONL_OUT="${PHYSTABMOL_INSTRUCTION_JSONL_OUT:-data/instruction_editing.jsonl}"
LIMIT="${PHYSTABMOL_LIMIT:-100000}"
MAX_PAIRS="${PHYSTABMOL_MAX_PAIRS:-50000}"
PAIRS_PER_SOURCE="${PHYSTABMOL_PAIRS_PER_SOURCE:-12}"
INSTRUCTIONS_PER_SPEC="${PHYSTABMOL_INSTRUCTIONS_PER_SPEC:-5}"
REFERENCE_POOL_SIZE="${PHYSTABMOL_REFERENCE_POOL_SIZE:-24}"
MIN_SIMILARITY="${PHYSTABMOL_MIN_SIMILARITY:-0.6}"
MAX_SIMILARITY="${PHYSTABMOL_MAX_SIMILARITY:-0.95}"
SEED="${PHYSTABMOL_SEED:-7}"

if [[ ! -s "$DATA" ]]; then
  echo "Dataset not found at $DATA; downloading ChEMBL first."
  PHYSTABMOL_OUT="$DATA" bash scripts/download_chembl_100k.sh --rdkit-filter
fi

python3 -m phystabmol.instruction_dataset \
  --data "$DATA" \
  --out "$OUT" \
  --jsonl-out "$JSONL_OUT" \
  --limit "$LIMIT" \
  --max-pairs "$MAX_PAIRS" \
  --pairs-per-source "$PAIRS_PER_SOURCE" \
  --instructions-per-spec "$INSTRUCTIONS_PER_SPEC" \
  --reference-pool-size "$REFERENCE_POOL_SIZE" \
  --min-similarity "$MIN_SIMILARITY" \
  --max-similarity "$MAX_SIMILARITY" \
  --seed "$SEED"

echo "Instruction dataset written to $OUT"
