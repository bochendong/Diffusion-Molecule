#!/bin/bash
set -euo pipefail

# Mine a reusable transform library for decoder_mode=hybrid_mmp.
#
# Usage from PhysTabMol repo root:
#   bash scripts/build_mmp_transform_library.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
export PHYSTABMOL_SUPPRESS_RDKIT_LOGS="${PHYSTABMOL_SUPPRESS_RDKIT_LOGS:-1}"
export PHYSTABMOL_PROGRESS="${PHYSTABMOL_PROGRESS:-1}"
export PHYSTABMOL_PROGRESS_STEP="${PHYSTABMOL_PROGRESS_STEP:-5}"
if command -v module >/dev/null 2>&1 && [[ "${PHYSTABMOL_USE_MODULE_ENV:-1}" == "1" ]]; then
  # Compute Canada/Nibi path: load Python/RDKit modules before activating the venv.
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_ROOT/scripts/env_module_venv.sh"
else
  # Local fallback for machines without environment modules.
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_ROOT/scripts/ensure_phystabmol_venv.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

"$PYTHON_BIN" - <<'PY'
from phystabmol import chem
print(f"rdkit_available={chem.RDKIT_AVAILABLE}")
if not chem.RDKIT_AVAILABLE:
    raise SystemExit("RDKit is required to mine fragment rows; source the module/venv environment first.")
PY

"$PYTHON_BIN" -m phystabmol.mmp_transform_library \
  --data "$DATA" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --max-pairs "$MAX_PAIRS" \
  --pairs-per-source "$PAIRS_PER_SOURCE" \
  --min-pair-similarity "$MIN_SIM" \
  --max-pair-similarity "$MAX_SIM" \
  --max-fragments "$MAX_FRAGMENTS" \
  --max-fragment-atoms "$MAX_FRAGMENT_ATOMS"

if [[ "${PHYSTABMOL_REQUIRE_MMP_FRAGMENTS:-1}" == "1" ]]; then
  if ! awk -F',' 'NR > 1 && $1 == "fragment" { found = 1; exit } END { exit found ? 0 : 1 }' "$OUT"; then
    cat <<EOF
No fragment rows were written to $OUT.
This benchmark needs fragment rows to avoid pair-only retrieval.
Try increasing PHYSTABMOL_MMP_MAX_FRAGMENT_ATOMS, or send this output back for debugging.
EOF
    exit 2
  fi
fi

echo "MMP transform library written to $OUT"
