#!/usr/bin/env bash
# Run SketchSMILES smoke tests.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES smoke"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"

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
  if [[ "${SKETCHSMILES_REQUIRE_RDKIT:-0}" == "1" ]]; then
    echo "ERROR: RDKit unavailable for $PYTHON_BIN." >&2
    echo "Try: SKETCHSMILES_MODULES=\"gcc rdkit/2025.09.4\" SKETCHSMILES_REQUIRE_RDKIT=1 bash scripts/run_smoke.sh" >&2
    exit 2
  fi
  echo "RDKit unavailable; skipping paired image render smoke."
  echo "Hint: on the server, try SKETCHSMILES_MODULES=\"gcc rdkit/2025.09.4\" bash scripts/run_smoke.sh"
fi
