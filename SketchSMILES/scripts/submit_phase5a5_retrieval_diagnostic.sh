#!/usr/bin/env bash
# Submit Phase 5A-5 retrieval diagnostic to a CPU Slurm node.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACCOUNT="${SKETCHSMILES_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHSMILES_SLURM_TIME:-02:00:00}"
MEM="${SKETCHSMILES_SLURM_MEM:-32G}"
CPUS="${SKETCHSMILES_SLURM_CPUS:-4}"
JOB_NAME="${SKETCHSMILES_SLURM_JOB_NAME:-sketchsmiles-5a5}"
LOG_DIR="${SKETCHSMILES_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export SKETCHSMILES_MODULES="${SKETCHSMILES_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHSMILES_PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
export SKETCHSMILES_SOURCE_RUN_DIR="${SKETCHSMILES_SOURCE_RUN_DIR:-outputs/runs/phase5a4_reranked_transformer_decoder_seed7}"
export SKETCHSMILES_RETRIEVAL_TOP_K="${SKETCHSMILES_RETRIEVAL_TOP_K:-16}"

echo "Submitting SketchSMILES Phase 5A-5 retrieval diagnostic:"
echo "  source_run_dir=$SKETCHSMILES_SOURCE_RUN_DIR"
echo "  python=$SKETCHSMILES_PYTHON_BIN"
echo "  retrieval_top_k=$SKETCHSMILES_RETRIEVAL_TOP_K"
echo "  max_eval=${SKETCHSMILES_MAX_EVAL:-<all>}"
echo "  slurm_time=$TIME"
echo "  slurm_mem=$MEM"
echo "  slurm_cpus=$CPUS"

sbatch \
  --account="$ACCOUNT" \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash 'scripts/run_phase5a5_retrieval_diagnostic.sh'"
