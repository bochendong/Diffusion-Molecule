#!/usr/bin/env bash
# Install PyTDC oracles into the MolEdit metrics venv on Alliance clusters.
#
# Usage:
#   module load gcc/12.3 rdkit/2024.09.6
#   SMU3M_PYTHON_BIN=/path/to/venv/bin/python bash scripts/setup_moledit_tdc.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}}"

if command -v module >/dev/null 2>&1 && ! "$PYTHON_BIN" -c "import rdkit" >/dev/null 2>&1; then
  echo "Loading gcc/12.3 rdkit/2024.09.6 for PyTDC install..."
  module load gcc/12.3 rdkit/2024.09.6
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "Installing PyTDC runtime deps into: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install PyTDC --no-deps
"$PYTHON_BIN" -m pip install \
  "fuzzywuzzy>=0.18" \
  python-Levenshtein \
  networkx \
  scipy \
  seaborn \
  scikit-learn \
  tqdm \
  "numpy<2" \
  "pandas<3"

echo "Verifying TDC oracles..."
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_moledit_table_metrics.py"
"$PYTHON_BIN" "$EVAL_SCRIPT" --help >/dev/null
"$PYTHON_BIN" - "$EVAL_SCRIPT" <<'PY'
import importlib.util
import sys
from pathlib import Path

eval_script = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("evaluate_moledit_table_metrics", eval_script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from tdc import Oracle

for name in ("GSK3B", "DRD2", "SA"):
    score = float(Oracle(name=name)("CCO"))
    print(f"  {name}: {score}")
print("PyTDC oracle check passed.")
PY
