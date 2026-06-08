#!/usr/bin/env bash
# Submit a packed generated + prior-only materialized benchmark for source-anchor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT="${SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceanchor_sweep_v1}"
export SMU3M_SOURCEANCHOR_BENCHMARK_LABEL="${SMU3M_SOURCEANCHOR_BENCHMARK_LABEL:-blend095_guard050_p005}"
export SMU3M_SOURCEANCHOR_BENCHMARK_MODES="${SMU3M_SOURCEANCHOR_BENCHMARK_MODES:-generated,prior}"
export SMU3M_BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SMMED_MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"

BENCH_ACCOUNT="${SMU3M_BENCHMARK_SLURM_ACCOUNT:-${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
BENCH_TIME="${SMU3M_BENCHMARK_SLURM_TIME:-00:45:00}"
BENCH_MEM="${SMU3M_BENCHMARK_SLURM_MEM:-4G}"
BENCH_CPUS="${SMU3M_BENCHMARK_SLURM_CPUS:-1}"
BENCH_JOB_NAME="${SMU3M_BENCHMARK_SLURM_JOB_NAME:-smu3m-srcanchor-prior}"
BENCH_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
BENCH_PARTITION="${SMU3M_BENCHMARK_SLURM_PARTITION:-${SMU3M_SLURM_PARTITION:-}}"

if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$BENCH_LOG_DIR"

echo "Submitting Unified 3M source-anchor prior benchmark"
echo "  sweep_output_root=$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT"
echo "  label=$SMU3M_SOURCEANCHOR_BENCHMARK_LABEL"
echo "  modes=$SMU3M_SOURCEANCHOR_BENCHMARK_MODES"
echo "  benchmark_profile=$SMU3M_BENCHMARK_PROFILE"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  bench_cpus=$BENCH_CPUS"
echo "  bench_mem=$BENCH_MEM"
echo "  bench_time=$BENCH_TIME"
echo "  resource_note=packed serial generated+prior benchmark; one CPU avoids multi-core underuse"

SBATCH_ARGS=(
  --account="$BENCH_ACCOUNT"
  --job-name="$BENCH_JOB_NAME"
  --time="$BENCH_TIME"
  --mem="$BENCH_MEM"
  --cpus-per-task="$BENCH_CPUS"
  --output="$BENCH_LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$BENCH_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$BENCH_PARTITION")
fi

submit_output="$(
  sbatch "${SBATCH_ARGS[@]}" \
    --wrap="bash '$PROJECT_DIR/scripts/run_unified_sourceanchor_prior_benchmark.sh'"
)"
echo "$submit_output"
job_id="$(echo "$submit_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to parse source-anchor benchmark job id." >&2
  exit 1
fi

output_dir="${SMU3M_SOURCEANCHOR_BENCHMARK_OUTPUT_DIR:-$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT/$SMU3M_SOURCEANCHOR_BENCHMARK_LABEL}"

echo
echo "Source-anchor benchmark submitted."
echo "  job_id=$job_id"
echo "  log=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-${job_id}.log"
echo "  generated_report=$output_dir/benchmark_materialized_${SMU3M_BENCHMARK_PROFILE}/benchmark_report.md"
echo "  prior_report=$output_dir/benchmark_materialized_prior_only_${SMU3M_BENCHMARK_PROFILE}/benchmark_report.md"
