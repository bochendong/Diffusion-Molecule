#!/usr/bin/env bash
# One-time: create PhysTabMol venv under scratch (Alliance: prefer module + venv over conda).
# Run on a login node from anywhere:
#   bash /path/to/PhysTabMol/scripts/setup_venv_phystabmol.sh
#
# RDKit: Alliance provides RDKit via environment modules only; `pip install rdkit` resolves to a
# dummy wheel that fails on purpose. Load an RDKit module *before* creating the venv, and use
# --system-site-packages so `import rdkit` sees the module build. Pick a version with:
#   module spider rdkit
# Override defaults, e.g.: MODULE_RDKIT=rdkit/2024.09.1 bash scripts/setup_venv_phystabmol.sh
#
# Torch: Alliance pip may inject +computecanada wheels; --isolated pulls official CUDA wheels.
# If isolated install fails, ask Alliance support for the recommended torch/cuda module pair.

set -euo pipefail

MODULE_GCC="${MODULE_GCC:-gcc}"
MODULE_PYTHON="${MODULE_PYTHON:-python/3.11.5}"
MODULE_RDKIT="${MODULE_RDKIT:-rdkit}"
VENV_DIR="${VENV_DIR:-/home/bdong/scratch/venvs/phystabmol}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSTABMOL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

module load "$MODULE_GCC" "$MODULE_PYTHON" "$MODULE_RDKIT"

python3 -m venv --system-site-packages "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel

python -m pip install --no-cache-dir \
  numpy pandas pillow scikit-learn

# Official PyTorch CUDA 12.4 wheels; change cu126 etc. per Alliance / driver docs.
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  --isolated \
  torch

python -c "import torch; print('torch', torch.__version__)"
python -c "import rdkit; print('rdkit ok')"
python -c "import sklearn; print('sklearn ok')"

echo "venv ready: $VENV_DIR"
echo "Each session (including Slurm): module load $MODULE_GCC $MODULE_PYTHON $MODULE_RDKIT && source $VENV_DIR/bin/activate"
echo "Project:  cd $PHYSTABMOL_ROOT"
