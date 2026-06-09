#!/usr/bin/env bash
# Submit sharded generated + prior-only materialized benchmarks for source-anchor.

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
export SMU3M_BASE_OUTPUT_DIR="${SMU3M_BASE_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_sourceguard_v1}"
export SMU3M_SOURCEANCHOR_BENCHMARK_LABEL="${SMU3M_SOURCEANCHOR_BENCHMARK_LABEL:-blend095_guard050_p005}"
export SMU3M_SOURCEANCHOR_BENCHMARK_MODES="${SMU3M_SOURCEANCHOR_BENCHMARK_MODES:-generated,prior}"
export SMU3M_BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
export SMU3M_BENCHMARK_SHARDS="${SMU3M_BENCHMARK_SHARDS:-5}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SMMED_MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-250}"
export SMMED_EVAL_SHARD_COUNT="$SMU3M_BENCHMARK_SHARDS"

BENCH_ACCOUNT="${SMU3M_BENCHMARK_SLURM_ACCOUNT:-${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
BENCH_TIME="${SMU3M_BENCHMARK_SLURM_TIME:-01:00:00}"
BENCH_MEM="${SMU3M_BENCHMARK_SLURM_MEM:-8G}"
BENCH_CPUS="${SMU3M_BENCHMARK_SLURM_CPUS:-1}"
BENCH_JOB_NAME="${SMU3M_BENCHMARK_SLURM_JOB_NAME:-smu3m-srcanchor}"
BENCH_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
BENCH_PARTITION="${SMU3M_BENCHMARK_SLURM_PARTITION:-${SMU3M_SLURM_PARTITION:-}}"

OUTPUT_DIR="${SMU3M_SOURCEANCHOR_BENCHMARK_OUTPUT_DIR:-$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT/$SMU3M_SOURCEANCHOR_BENCHMARK_LABEL}"

if (( SMU3M_BENCHMARK_SHARDS <= 0 )); then
  echo "ERROR: SMU3M_BENCHMARK_SHARDS must be positive, got $SMU3M_BENCHMARK_SHARDS" >&2
  exit 2
fi
if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$BENCH_LOG_DIR"

echo "Submitting Unified 3M source-anchor materialized benchmark (sharded)"
echo "  sweep_output_root=$SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT"
echo "  base_output_dir=$SMU3M_BASE_OUTPUT_DIR"
echo "  label=$SMU3M_SOURCEANCHOR_BENCHMARK_LABEL"
echo "  output_dir=$OUTPUT_DIR"
echo "  modes=$SMU3M_SOURCEANCHOR_BENCHMARK_MODES"
echo "  benchmark_profile=$SMU3M_BENCHMARK_PROFILE"
echo "  benchmark_shards=$SMU3M_BENCHMARK_SHARDS"
echo "  max_eval_per_property_count=$SMMED_MAX_EVAL_PER_PROPERTY_COUNT"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  bench_cpus=$BENCH_CPUS"
echo "  bench_mem=$BENCH_MEM"
echo "  bench_time=$BENCH_TIME (per shard)"

SBATCH_ARGS=(
  --account="$BENCH_ACCOUNT"
  --job-name="$BENCH_JOB_NAME"
  --time="$BENCH_TIME"
  --mem="$BENCH_MEM"
  --cpus-per-task="$BENCH_CPUS"
  --export=ALL
)
if [[ -n "$BENCH_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$BENCH_PARTITION")
fi

bench_job_ids=()
IFS=',' read -r -a modes <<< "$SMU3M_SOURCEANCHOR_BENCHMARK_MODES"
if (( ${#modes[@]} == 0 )); then
  echo "ERROR: no benchmark modes requested." >&2
  exit 2
fi

for raw_mode in "${modes[@]}"; do
  mode="$(printf '%s' "$raw_mode" | xargs)"
  [[ -z "$mode" ]] && continue
  case "$mode" in
    generated) mode_tag="gen" ;;
    prior | prior_only) mode_tag="prior" ;;
    *) mode_tag="$mode" ;;
  esac

  for shard_index in $(seq 0 "$((SMU3M_BENCHMARK_SHARDS - 1))"); do
    shard_output="$(
      sbatch "${SBATCH_ARGS[@]}" \
        --output="$BENCH_LOG_DIR/${BENCH_JOB_NAME}-${mode_tag}-s${shard_index}-%j.log" \
        --wrap="SMU3M_SOURCEANCHOR_BENCHMARK_MODES=$mode SMMED_EVAL_SHARD_INDEX=$shard_index bash '$PROJECT_DIR/scripts/run_unified_sourceanchor_prior_benchmark.sh'"
    )"
    echo "$shard_output"
    shard_job_id="$(echo "$shard_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ -z "$shard_job_id" ]]; then
      echo "ERROR: failed to parse shard job id for mode=$mode shard=$shard_index." >&2
      exit 1
    fi
    bench_job_ids+=("$shard_job_id")
  done
done

echo
echo "Source-anchor benchmark shards submitted."
echo "  job_ids=${bench_job_ids[*]}"
echo "  logs=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-<mode>-s<shard>-<job_id>.log"
echo
echo "After all shards finish, merge each mode:"
for raw_mode in "${modes[@]}"; do
  mode="$(printf '%s' "$raw_mode" | xargs)"
  [[ -z "$mode" ]] && continue
  case "$mode" in
    generated)
      bench_dir="$OUTPUT_DIR/benchmark_materialized_${SMU3M_BENCHMARK_PROFILE}"
      ;;
    prior | prior_only)
      bench_dir="$OUTPUT_DIR/benchmark_materialized_prior_only_${SMU3M_BENCHMARK_PROFILE}"
      ;;
    *)
      continue
      ;;
  esac
  echo "  $mode:"
  echo "    SMU3M_OUTPUT_DIR=$OUTPUT_DIR SMU3M_BENCHMARK_OUTPUT_DIR=$bench_dir SMU3M_BENCHMARK_PROFILE=$SMU3M_BENCHMARK_PROFILE SMU3M_PYTHON_BIN=$SMU3M_PYTHON_BIN bash $PROJECT_DIR/scripts/merge_unified_materialized_benchmark_shards.sh"
  echo "    report=$bench_dir/benchmark_report.md"
done
