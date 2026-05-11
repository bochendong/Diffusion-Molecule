#!/bin/bash
# Submit the SketchMol-comparable PhysTabMol benchmark with structure prompts enabled.
#
# Usage from the PhysTabMol repo root:
#   bash scripts/run_sketchmol_structure_benchmark.sh
#
# Optional overrides:
#   PHYSTABMOL_RUN_NAME=my_run PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS=64 bash scripts/run_sketchmol_structure_benchmark.sh
#   bash scripts/run_sketchmol_structure_benchmark.sh --time=20:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_NAME:-sketchmol_comparable_structure_v1}"
export PHYSTABMOL_DECODER_MODE="${PHYSTABMOL_DECODER_MODE:-hybrid_mmp}"
export PHYSTABMOL_RETRIEVAL_EXACT_PENALTY="${PHYSTABMOL_RETRIEVAL_EXACT_PENALTY:-0.25}"
export PHYSTABMOL_RETRIEVAL_EDIT_BONUS="${PHYSTABMOL_RETRIEVAL_EDIT_BONUS:-0.15}"
export PHYSTABMOL_MMP_MAX_PAIRS="${PHYSTABMOL_MMP_MAX_PAIRS:-80000}"
export PHYSTABMOL_MMP_FRAGMENT_NEIGHBORS="${PHYSTABMOL_MMP_FRAGMENT_NEIGHBORS:-12}"
export PHYSTABMOL_MMP_EXACT_PENALTY="${PHYSTABMOL_MMP_EXACT_PENALTY:-0.75}"
export PHYSTABMOL_MMP_FRAGMENT_BONUS="${PHYSTABMOL_MMP_FRAGMENT_BONUS:-0.24}"
export PHYSTABMOL_MMP_FRAGMENT_EXACT_PENALTY="${PHYSTABMOL_MMP_FRAGMENT_EXACT_PENALTY:-0.30}"
export PHYSTABMOL_PROPERTY_MASK_CONDITIONING="${PHYSTABMOL_PROPERTY_MASK_CONDITIONING:-1}"
export PHYSTABMOL_RUN_SKETCHMOL_BENCHMARK="${PHYSTABMOL_RUN_SKETCHMOL_BENCHMARK:-1}"
export PHYSTABMOL_RUN_STRUCTURE_PROMPT_BENCHMARK="${PHYSTABMOL_RUN_STRUCTURE_PROMPT_BENCHMARK:-1}"

export PHYSTABMOL_TORCH_BATCH_SIZE="${PHYSTABMOL_TORCH_BATCH_SIZE:-4096}"
export PHYSTABMOL_SAMPLE_CHUNK_SIZE="${PHYSTABMOL_SAMPLE_CHUNK_SIZE:-8192}"

export PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS="${PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS:-125}"
export PHYSTABMOL_BENCHMARK_SAMPLES="${PHYSTABMOL_BENCHMARK_SAMPLES:-8}"
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

if [[ -s data/mmp_transform_library.csv ]] && [[ "${PHYSTABMOL_ALLOW_FRAGMENTLESS_LIBRARY:-0}" != "1" ]]; then
  if ! awk -F',' 'NR > 1 && $1 == "fragment" { found = 1; exit } END { exit found ? 0 : 1 }' data/mmp_transform_library.csv; then
    cat <<'EOF'
Existing data/mmp_transform_library.csv has no fragment rows.
Rebuild it before submitting the publication benchmark:
  bash scripts/build_mmp_transform_library.sh

To intentionally run the old pair-only library, set:
  PHYSTABMOL_ALLOW_FRAGMENTLESS_LIBRARY=1
EOF
    exit 2
  fi
fi

cat <<EOF
Submitting PhysTabMol SketchMol-comparable structure benchmark:
  run_name=$PHYSTABMOL_RUN_NAME
  decoder_mode=$PHYSTABMOL_DECODER_MODE
  retrieval_exact_penalty=$PHYSTABMOL_RETRIEVAL_EXACT_PENALTY
  retrieval_edit_bonus=$PHYSTABMOL_RETRIEVAL_EDIT_BONUS
  mmp_max_pairs=$PHYSTABMOL_MMP_MAX_PAIRS
  mmp_fragment_neighbors=$PHYSTABMOL_MMP_FRAGMENT_NEIGHBORS
  mmp_exact_penalty=$PHYSTABMOL_MMP_EXACT_PENALTY
  mmp_fragment_bonus=$PHYSTABMOL_MMP_FRAGMENT_BONUS
  property_mask_conditioning=$PHYSTABMOL_PROPERTY_MASK_CONDITIONING
  sketchmol_benchmark=$PHYSTABMOL_RUN_SKETCHMOL_BENCHMARK
  structure_prompt_benchmark=$PHYSTABMOL_RUN_STRUCTURE_PROMPT_BENCHMARK
  torch_batch_size=$PHYSTABMOL_TORCH_BATCH_SIZE
  benchmark_single_conditions=$PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS
  benchmark_samples_per_condition=$PHYSTABMOL_BENCHMARK_SAMPLES
  benchmark_total_per_single_target=$((PHYSTABMOL_BENCHMARK_SINGLE_CONDITIONS * PHYSTABMOL_BENCHMARK_SAMPLES))
  structure_prompt_conditions=$PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS
EOF

SBATCH_ARGS=("--time=${PHYSTABMOL_WALLTIME:-20:00:00}")
sbatch --export=ALL "${SBATCH_ARGS[@]}" "$@" scripts/run_phystabmol_gpu.slurm.sh
