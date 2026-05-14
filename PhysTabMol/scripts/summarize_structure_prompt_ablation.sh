#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$ROOT}"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/ensure_phystabmol_venv.sh"

RUN_DIR="${PHYSTABMOL_RUN_DIR:-$(find runs -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m phystabmol.structure_ablation_summary --run-dir "$RUN_DIR" "$@"
