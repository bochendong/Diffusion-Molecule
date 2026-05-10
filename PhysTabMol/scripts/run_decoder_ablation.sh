#!/bin/bash
set -euo pipefail

# Submit comparable decoder ablations. By default this runs four jobs:
# physics, retrieval, hybrid, hybrid_mmp.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODES="${PHYSTABMOL_DECODER_ABLATION_MODES:-physics retrieval hybrid hybrid_mmp}"

for mode in $MODES; do
  export PHYSTABMOL_DECODER_MODE="$mode"
  export PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_PREFIX:-decoder_ablation}_${mode}"
  echo "Submitting decoder ablation: mode=$PHYSTABMOL_DECODER_MODE run_name=$PHYSTABMOL_RUN_NAME"
  bash scripts/run_sketchmol_structure_benchmark.sh "$@"
done
