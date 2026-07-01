#!/usr/bin/env bash
# Create a dedicated venv for ADMET-AI oracle scoring (ComputeCanada-friendly).
#
# Usage:
#   bash SketchMol-Understanding-Condition/scripts/setup_admet_ai_venv.sh
#
# Then:
#   export SUCC_ADMET_PYTHON_BIN=$HOME/.venvs/admet_ai/bin/python

set -euo pipefail

VENV_DIR="${SUCC_ADMET_VENV_DIR:-$HOME/.venvs/admet_ai}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "ADMET-AI venv setup"
echo "  venv_dir=$VENV_DIR"
echo "  python=$PYTHON_BIN ($(command -v "$PYTHON_BIN"))"

if [[ -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi

"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# admet-ai declares pip-installable rdkit, which is a dummy wheel on ComputeCanada.
# RDKit comes from the loaded module via --system-site-packages.
python -m pip install admet-ai --no-deps
python -m pip install \
  "chemprop>=2.2.2" \
  "typed-argument-parser>=1.11.0" \
  seaborn tqdm lightning pandas numpy

python - <<'PY'
import rdkit
from rdkit import Chem
print("rdkit ok", rdkit.__version__, Chem.MolFromSmiles("CCO") is not None)
PY

python - <<'PY'
from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles=["CCO", "c1ccccc1"])
print("admet_ai smoke ok", list(preds.columns)[:8])
PY

echo
echo "Done."
echo "  export SUCC_ADMET_PYTHON_BIN=$VENV_DIR/bin/python"
