#!/usr/bin/env bash
# One-time setup for a SketchImage-JEPA venv with RDKit + CUDA PyTorch.
#
# Run on a login node:
#   MODULE_RDKIT=rdkit/2025.09.4 VENV_DIR=/scratch/bdong/venvs/sketchimage-rdkit \
#     bash scripts/setup_torch_venv.sh

set -euo pipefail

MODULE_GCC="${MODULE_GCC:-gcc}"
MODULE_PYTHON="${MODULE_PYTHON:-python/3.11.5}"
MODULE_RDKIT="${MODULE_RDKIT:-rdkit}"
MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"
VENV_DIR="${VENV_DIR:-/scratch/bdong/venvs/sketchimage-rdkit}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if command -v module >/dev/null 2>&1; then
  module load "$MODULE_GCC" "$MODULE_PYTHON" "$MODULE_RDKIT"
  if [[ -n "$MODULE_CUDA" ]]; then
    module load "$MODULE_CUDA"
  fi
else
  echo "WARNING: module command not found; continuing with current shell environment." >&2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir numpy pandas pillow
python -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" --isolated torch

python - <<'PY'
import sys
import numpy
import torch

print("python=", sys.executable)
print("numpy=", numpy.__version__)
print("torch=", torch.__version__)
print("torch_cuda_available=", torch.cuda.is_available())
PY

python - <<'PY'
try:
    import rdkit
    print("rdkit=", rdkit.__version__)
except Exception as exc:
    raise SystemExit(f"ERROR: RDKit is still unavailable in this venv: {exc}")
PY

echo "venv ready: $VENV_DIR"
