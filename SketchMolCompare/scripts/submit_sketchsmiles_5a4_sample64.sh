#!/usr/bin/env bash
# Submit a stochastic candidate-generation Phase 5A-4 run with fingerprint reranking.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SKETCHMOL_COMPARE_SEED:-${SKETCHSMILES_SEED:-7}}"

export SKETCHSMILES_SEED="$SEED"
export SKETCHSMILES_RUN_NAME="${SKETCHSMILES_RUN_NAME:-sketchmol_compare_phase5a4_sample64_seed${SEED}}"
export SKETCHSMILES_DECODING="${SKETCHSMILES_DECODING:-sample}"
export SKETCHSMILES_SAMPLES_PER_CONDITION="${SKETCHSMILES_SAMPLES_PER_CONDITION:-64}"
export SKETCHSMILES_SAMPLE_TOP_K="${SKETCHSMILES_SAMPLE_TOP_K:-32}"
export SKETCHSMILES_TEMPERATURE="${SKETCHSMILES_TEMPERATURE:-0.95}"
export SKETCHSMILES_SLURM_JOB_NAME="${SKETCHSMILES_SLURM_JOB_NAME:-sketchsmiles-5a4-s64}"
export SKETCHSMILES_SLURM_TIME="${SKETCHSMILES_SLURM_TIME:-12:00:00}"

echo "SketchMolCompare -> SketchSMILES Phase 5A-4 sample64"
echo "  run_name=$SKETCHSMILES_RUN_NAME"
echo "  decoding=$SKETCHSMILES_DECODING"
echo "  samples_per_condition=$SKETCHSMILES_SAMPLES_PER_CONDITION"
echo "  sample_top_k=$SKETCHSMILES_SAMPLE_TOP_K"
echo "  temperature=$SKETCHSMILES_TEMPERATURE"

bash "$SCRIPT_DIR/submit_sketchsmiles_5a4_transformer.sh" "$@"
