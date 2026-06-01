#!/usr/bin/env bash
# Submit the PhysTabMol SketchMol-aligned benchmark from the comparison folder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PHYSTABMOL_DIR="$REPO_ROOT/PhysTabMol"

if [[ ! -d "$PHYSTABMOL_DIR" ]]; then
  echo "ERROR: PhysTabMol folder not found at $PHYSTABMOL_DIR" >&2
  exit 2
fi

cd "$PHYSTABMOL_DIR"

SEED="${SKETCHMOL_COMPARE_SEED:-7}"
export PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_NAME:-sketchmol_compare_structure_seed${SEED}}"
export PHYSTABMOL_WALLTIME="${PHYSTABMOL_WALLTIME:-20:00:00}"
export PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS="${PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS:-125}"
export PHYSTABMOL_BENCHMARK_SAMPLES="${PHYSTABMOL_BENCHMARK_SAMPLES:-8}"
export PHYSTABMOL_BENCHMARK_MULTI_CONDITIONS="${PHYSTABMOL_BENCHMARK_MULTI_CONDITIONS:-1000}"
export PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS="${PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS:-1000}"
export PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES="${PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES:-8}"

echo "SketchMolCompare -> PhysTabMol SketchMol-aligned benchmark"
echo "  cwd=$PWD"
echo "  run_name=$PHYSTABMOL_RUN_NAME"
echo "  walltime=$PHYSTABMOL_WALLTIME"

bash scripts/run_sketchmol_structure_benchmark.sh "$@"
