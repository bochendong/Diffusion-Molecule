#!/usr/bin/env bash
# Run SketchSMILES smoke tests.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

echo "SketchSMILES smoke"
echo "  python=$PYTHON_BIN"

"$PYTHON_BIN" -m unittest discover -s tests

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import rdkit  # noqa: F401
PY
then
  "$PYTHON_BIN" -m sketch_smiles.build_pairs \
    --input-csv data/example_molecules.csv \
    --output-dir outputs/pairs/smoke \
    --limit 5
else
  echo "RDKit unavailable; skipping paired image render smoke."
fi
