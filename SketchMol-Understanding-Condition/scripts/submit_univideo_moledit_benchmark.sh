#!/usr/bin/env bash
# Submit SUCC UniVideo MolEdit materialized benchmark and optional table metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SUCC_DATASET_MODE=moledit
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v1}"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
export SUCC_MATERIALIZED_SLURM_TIME="${SUCC_MATERIALIZED_SLURM_TIME:-04:00:00}"
export SUCC_MATERIALIZED_SLURM_MEM="${SUCC_MATERIALIZED_SLURM_MEM:-16G}"
export SUCC_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK="${SUCC_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK:-1}"

case "$SUCC_MATERIALIZED_BENCHMARK_PROFILE" in
  primary_fast) DEFAULT_BENCHMARK_OUTPUT_DIR="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_primary_fast" ;;
  oracle) DEFAULT_BENCHMARK_OUTPUT_DIR="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_oracle" ;;
  latent) DEFAULT_BENCHMARK_OUTPUT_DIR="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_latent" ;;
  *) DEFAULT_BENCHMARK_OUTPUT_DIR="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_${SUCC_MATERIALIZED_BENCHMARK_PROFILE}" ;;
esac
BENCHMARK_OUTPUT_DIR="${SUCC_MATERIALIZED_BENCHMARK_OUTPUT_DIR:-$DEFAULT_BENCHMARK_OUTPUT_DIR}"
export SUCC_MOLEDIT_TABLE_PREDICTIONS="${SUCC_MOLEDIT_TABLE_PREDICTIONS:-$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv}"
export SUCC_MOLEDIT_TABLE_OUTPUT_DIR="${SUCC_MOLEDIT_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics}"

echo "Submitting SUCC UniVideo MolEdit benchmark workflow"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  benchmark_profile=$SUCC_MATERIALIZED_BENCHMARK_PROFILE"
echo "  table_after_benchmark=$SUCC_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK"
echo

benchmark_output="$(bash "$SCRIPT_DIR/submit_univideo_materialized_benchmark.sh")"
echo "$benchmark_output"
benchmark_job_ids=()
while IFS= read -r benchmark_job_id; do
  if [[ -n "$benchmark_job_id" ]]; then
    benchmark_job_ids+=("$benchmark_job_id")
  fi
done < <(echo "$benchmark_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')

if [[ "$SUCC_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK" != "1" ]]; then
  echo
  echo "MolEdit table metrics submission skipped."
  echo "  next=bash $SCRIPT_DIR/submit_univideo_moledit_table_metrics.sh"
  exit 0
fi

if (( ${#benchmark_job_ids[@]} == 0 )); then
  echo "ERROR: failed to parse benchmark job id for MolEdit table dependency." >&2
  exit 1
fi

benchmark_dependency="afterok:$(IFS=:; echo "${benchmark_job_ids[*]}")"
echo
echo "Submitting dependent MolEdit table metrics"
echo "  dependency=$benchmark_dependency"
SUCC_TABLE_SLURM_DEPENDENCY="$benchmark_dependency" \
  bash "$SCRIPT_DIR/submit_univideo_moledit_table_metrics.sh"
