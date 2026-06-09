#!/usr/bin/env bash
# Submit the Unified 3M MolEdit materialized benchmark and optional table metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SMU3M_DATASET_MODE=moledit
export SMU3M_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SMU3M_MOLEDIT_EVAL_SPLIT="${SMU3M_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SMU3M_MOLEDIT_CONDITION_ROWS="${SMU3M_MOLEDIT_CONDITION_ROWS:-$SMU3M_OUTPUT_DIR/dataset/moledit_benchmark_condition_rows.csv}"
export SMMED_CONDITION_ROWS="${SMMED_CONDITION_ROWS:-$SMU3M_MOLEDIT_CONDITION_ROWS}"
export SMU3M_BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
export SMU3M_BENCHMARK_SHARDS="${SMU3M_BENCHMARK_SHARDS:-1}"
export SMU3M_BENCHMARK_SUBMIT_MODE="${SMU3M_BENCHMARK_SUBMIT_MODE:-jobs}"
export SMU3M_BENCHMARK_SLURM_TIME="${SMU3M_BENCHMARK_SLURM_TIME:-04:00:00}"
export SMU3M_BENCHMARK_SLURM_MEM="${SMU3M_BENCHMARK_SLURM_MEM:-16G}"
export SMU3M_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK="${SMU3M_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK:-1}"

case "$SMU3M_BENCHMARK_PROFILE" in
  full) DEFAULT_BENCHMARK_OUTPUT_DIR="$SMU3M_OUTPUT_DIR/benchmark_materialized" ;;
  *) DEFAULT_BENCHMARK_OUTPUT_DIR="$SMU3M_OUTPUT_DIR/benchmark_materialized_${SMU3M_BENCHMARK_PROFILE}" ;;
esac
BENCHMARK_OUTPUT_DIR="${SMU3M_BENCHMARK_OUTPUT_DIR:-$DEFAULT_BENCHMARK_OUTPUT_DIR}"
export SMU3M_MOLEDIT_TABLE_PREDICTIONS="${SMU3M_MOLEDIT_TABLE_PREDICTIONS:-$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv}"
export SMU3M_MOLEDIT_TABLE_OUTPUT_DIR="${SMU3M_MOLEDIT_TABLE_OUTPUT_DIR:-$SMU3M_OUTPUT_DIR/moledit_table_metrics}"

echo "Submitting Unified 3M MolEdit benchmark workflow"
echo "  output_dir=$SMU3M_OUTPUT_DIR"
echo "  condition_rows=$SMMED_CONDITION_ROWS"
echo "  benchmark_profile=$SMU3M_BENCHMARK_PROFILE"
echo "  benchmark_shards=$SMU3M_BENCHMARK_SHARDS"
echo "  benchmark_time=$SMU3M_BENCHMARK_SLURM_TIME"
echo "  submit_table_after_benchmark=$SMU3M_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK"
echo

benchmark_output="$(bash "$SCRIPT_DIR/submit_unified_materialized_benchmark.sh")"
echo "$benchmark_output"
benchmark_job_ids=()
while IFS= read -r benchmark_job_id; do
  if [[ -n "$benchmark_job_id" ]]; then
    benchmark_job_ids+=("$benchmark_job_id")
  fi
done < <(echo "$benchmark_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')

if [[ "$SMU3M_SUBMIT_MOLEDIT_TABLE_AFTER_BENCHMARK" != "1" ]]; then
  echo
  echo "MolEdit table metrics submission skipped."
  echo "  next=bash $SCRIPT_DIR/submit_unified_moledit_table_metrics.sh"
  exit 0
fi

if (( SMU3M_BENCHMARK_SHARDS != 1 )); then
  echo
  echo "MolEdit table metrics submission skipped for sharded benchmark."
  echo "  merge first, then run: bash $SCRIPT_DIR/submit_unified_moledit_table_metrics.sh"
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
SMU3M_TABLE_SLURM_DEPENDENCY="$benchmark_dependency" \
  bash "$SCRIPT_DIR/submit_unified_moledit_table_metrics.sh"
