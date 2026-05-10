#!/bin/bash
set -euo pipefail

# Mine a reusable transform library for decoder_mode=hybrid_mmp.
#
# Usage from PhysTabMol repo root:
#   bash scripts/build_mmp_transform_library.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
# shellcheck source=/dev/null
source "$PHYSTABMOL_ROOT/scripts/ensure_phystabmol_venv.sh"

DATA="${PHYSTABMOL_DATA:-data/molecules.csv}"
OUT="${PHYSTABMOL_MMP_TRANSFORM_LIBRARY:-data/mmp_transform_library.csv}"
LIMIT="${PHYSTABMOL_LIMIT:-100000}"
MAX_PAIRS="${PHYSTABMOL_MMP_MAX_PAIRS:-80000}"
PAIRS_PER_SOURCE="${PHYSTABMOL_MMP_PAIRS_PER_SOURCE:-6}"
MIN_SIM="${PHYSTABMOL_MMP_MIN_PAIR_SIMILARITY:-0.25}"
MAX_SIM="${PHYSTABMOL_MMP_MAX_PAIR_SIMILARITY:-0.98}"
MAX_FRAGMENTS="${PHYSTABMOL_MMP_MAX_FRAGMENTS:-12000}"
MAX_FRAGMENT_ATOMS="${PHYSTABMOL_MMP_MAX_FRAGMENT_ATOMS:-8}"

if [[ ! -s "$DATA" ]]; then
  echo "Dataset not found at $DATA; downloading ChEMBL first."
  PHYSTABMOL_OUT="$DATA" bash scripts/download_chembl_100k.sh --rdkit-filter
fi

python3 -m phystabmol.mmp_transform_library \
  --data "$DATA" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --max-pairs "$MAX_PAIRS" \
  --pairs-per-source "$PAIRS_PER_SOURCE" \
  --min-pair-similarity "$MIN_SIM" \
  --max-pair-similarity "$MAX_SIM" \
  --max-fragments "$MAX_FRAGMENTS" \
  --max-fragment-atoms "$MAX_FRAGMENT_ATOMS"

echo "MMP transform library written to $OUT"
