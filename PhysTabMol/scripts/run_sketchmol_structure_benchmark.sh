#!/bin/bash
# Submit the SketchMol-comparable PhysTabMol benchmark with structure prompts enabled.
#
# Usage from the PhysTabMol repo root:
#   bash scripts/run_sketchmol_structure_benchmark.sh
#
# Optional overrides:
#   PHYSTABMOL_RUN_NAME=my_run PHYSTABMOL_BENCHMARK_SAMPLES=500 bash scripts/run_sketchmol_structure_benchmark.sh
#   bash scripts/run_sketchmol_structure_benchmark.sh --time=20:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_NAME:-sketchmol_comparable_structure_v1}"
export PHYSTABMOL_PROPERTY_MASK_CONDITIONING="${PHYSTABMOL_PROPERTY_MASK_CONDITIONING:-1}"
export PHYSTABMOL_RUN_SKETCHMOL_BENCHMARK="${PHYSTABMOL_RUN_SKETCHMOL_BENCHMARK:-1}"
export PHYSTABMOL_RUN_STRUCTURE_PROMPT_BENCHMARK="${PHYSTABMOL_RUN_STRUCTURE_PROMPT_BENCHMARK:-1}"

export PHYSTABMOL_BENCHMARK_SAMPLES="${PHYSTABMOL_BENCHMARK_SAMPLES:-1000}"
export PHYSTABMOL_BENCHMARK_MULTI_CONDITIONS="${PHYSTABMOL_BENCHMARK_MULTI_CONDITIONS:-1000}"
export PHYSTABMOL_BENCHMARK_OPTIMIZATION_CONDITIONS="${PHYSTABMOL_BENCHMARK_OPTIMIZATION_CONDITIONS:-100}"
export PHYSTABMOL_BENCHMARK_DECODE_TOP_K="${PHYSTABMOL_BENCHMARK_DECODE_TOP_K:-1}"

export PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS="${PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS:-1000}"
export PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES="${PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES:-8}"
export PHYSTABMOL_STRUCTURE_PROMPT_DECODE_TOP_K="${PHYSTABMOL_STRUCTURE_PROMPT_DECODE_TOP_K:-2}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on a Slurm login node, or call scripts/run_phystabmol_gpu.slurm.sh directly inside an allocation."
  exit 127
fi

cat <<EOF
Submitting PhysTabMol SketchMol-comparable structure benchmark:
  run_name=$PHYSTABMOL_RUN_NAME
  property_mask_conditioning=$PHYSTABMOL_PROPERTY_MASK_CONDITIONING
  sketchmol_benchmark=$PHYSTABMOL_RUN_SKETCHMOL_BENCHMARK
  structure_prompt_benchmark=$PHYSTABMOL_RUN_STRUCTURE_PROMPT_BENCHMARK
  benchmark_samples=$PHYSTABMOL_BENCHMARK_SAMPLES
  structure_prompt_conditions=$PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS
EOF

sbatch --export=ALL "$@" scripts/run_phystabmol_gpu.slurm.sh
