#!/usr/bin/env bash
# Run lightweight tests for Latent Edit Trajectory Attention.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LATENT_EDIT_TRAJECTORY_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Latent Edit Trajectory Attention smoke"
echo "  python=$PYTHON_BIN"

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'
if "$PYTHON_BIN" - <<'PY'
try:
    import torch  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
then
  "$PYTHON_BIN" -m latent_edit_trajectory_attention.train \
    --output-dir outputs/runs/smoke_synthetic \
    --examples 24 \
    --history-length 4 \
    --latent-dim 16 \
    --property-dim 2 \
    --target-dim 2 \
    --hidden-dim 32 \
    --transformer-layers 1 \
    --attention-heads 4 \
    --diffusion-steps 8 \
    --max-history 4 \
    --epochs 1 \
    --batch-size 8 \
    --device cpu
else
  echo "PyTorch is not installed; skipped synthetic training smoke."
fi

