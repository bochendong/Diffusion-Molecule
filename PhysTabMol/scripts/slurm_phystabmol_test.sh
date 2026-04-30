#!/usr/bin/env bash
# Run inside an interactive allocation (salloc) or batch job to test PhysTabMol.
# Uses Alliance-style module + venv (see setup_venv_phystabmol.sh).

set -euo pipefail

unset LD_LIBRARY_PATH
unset PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_module_venv.sh"

# Mode: "smoke" (fast) | "full" (README-style defaults, heavier)
MODE="${1:-smoke}"

echo "host=$(hostname) mode=$MODE cuda_visible=${CUDA_VISIBLE_DEVICES:-}"

if [[ "$MODE" == "full" ]]; then
  python -m phystabmol.experiment \
    --backend auto \
    --run-name salloc_full \
    --samples-per-condition 32 \
    --decode-top-k 5
else
  python -m phystabmol.experiment \
    --backend sklearn \
    --contrastive-epochs 30 \
    --run-name salloc_smoke \
    --samples-per-condition 8 \
    --decode-top-k 3 \
    --timesteps 40 \
    --noise-repeats 8
fi

echo "Done."
