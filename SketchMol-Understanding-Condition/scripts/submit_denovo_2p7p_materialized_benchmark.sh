#!/usr/bin/env bash
# Submit the de novo 2p-7p property-design benchmark as a Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_DENOVO_OUTPUT_DIR="${SUCC_DENOVO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/denovo_2p7p_v1}"
export SUCC_DENOVO_MOLECULE_DB_CSV="${SUCC_DENOVO_MOLECULE_DB_CSV:-$SMMED_DEFAULT_MOLECULE_DB}"
export SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT="${SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT:-1000}"
export SUCC_DENOVO_MATERIALIZED_METHODS="${SUCC_DENOVO_MATERIALIZED_METHODS:-property_nearest,target_oracle}"

BENCH_ACCOUNT="${SUCC_DENOVO_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
BENCH_TIME="${SUCC_DENOVO_SLURM_TIME:-${SUCC_SLURM_TIME:-02:00:00}}"
BENCH_MEM="${SUCC_DENOVO_SLURM_MEM:-${SUCC_SLURM_MEM:-16G}}"
BENCH_CPUS="${SUCC_DENOVO_SLURM_CPUS:-${SUCC_SLURM_CPUS:-1}}"
BENCH_JOB_NAME="${SUCC_DENOVO_SLURM_JOB_NAME:-succ-denovo-2p7p}"
BENCH_LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
BENCH_PARTITION="${SUCC_DENOVO_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$SUCC_DENOVO_MOLECULE_DB_CSV" ]]; then
  echo "ERROR: missing molecule database: $SUCC_DENOVO_MOLECULE_DB_CSV" >&2
  exit 2
fi

mkdir -p "$BENCH_LOG_DIR"

echo "Submitting de novo 2p-7p materialized benchmark"
echo "  output_dir=$SUCC_DENOVO_OUTPUT_DIR"
echo "  molecule_db=$SUCC_DENOVO_MOLECULE_DB_CSV"
echo "  rows_per_property_count=$SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT"
echo "  methods=$SUCC_DENOVO_MATERIALIZED_METHODS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  bench_cpus=$BENCH_CPUS"
echo "  bench_mem=$BENCH_MEM"
echo "  bench_time=$BENCH_TIME"

BENCH_SBATCH_ARGS=(
  --account="$BENCH_ACCOUNT"
  --job-name="$BENCH_JOB_NAME"
  --time="$BENCH_TIME"
  --mem="$BENCH_MEM"
  --cpus-per-task="$BENCH_CPUS"
  --export=ALL
  --output="$BENCH_LOG_DIR/%x-%j.log"
)
if [[ -n "$BENCH_PARTITION" ]]; then
  BENCH_SBATCH_ARGS+=(--partition="$BENCH_PARTITION")
fi

bench_output="$(
  sbatch "${BENCH_SBATCH_ARGS[@]}" \
    --wrap="bash '$PROJECT_DIR/scripts/run_denovo_2p7p_materialized_benchmark.sh'"
)"
echo "$bench_output"
bench_job_id="$(echo "$bench_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$bench_job_id" ]]; then
  echo "ERROR: failed to parse benchmark job id." >&2
  exit 1
fi

echo
echo "De novo 2p-7p benchmark submitted."
echo "  job_id=$bench_job_id"
echo "  log=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-${bench_job_id}.log"
echo "  benchmark_report=$SUCC_DENOVO_OUTPUT_DIR/benchmark_materialized/benchmark_report.md"
