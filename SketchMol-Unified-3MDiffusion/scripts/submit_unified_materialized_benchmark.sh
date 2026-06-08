#!/usr/bin/env bash
# Submit Unified 3M materialized benchmark as a separate Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SMU3M_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_v1}"
export SMMED_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1}"
export SMU3M_PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SMU3M_BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
export SMU3M_BENCHMARK_SHARDS="${SMU3M_BENCHMARK_SHARDS:-1}"
export SMU3M_BENCHMARK_SUBMIT_MODE="${SMU3M_BENCHMARK_SUBMIT_MODE:-jobs}"
export SMU3M_SOURCE_SIMILARITY_RERANK_CANDIDATES="${SMU3M_SOURCE_SIMILARITY_RERANK_CANDIDATES:-256}"
export SMU3M_RESTRICT_BENCHMARK_TO_EDIT_LATENT_INDEX="${SMU3M_RESTRICT_BENCHMARK_TO_EDIT_LATENT_INDEX:-1}"
export SMMED_MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"
export SMMED_EVAL_SHARD_COUNT="$SMU3M_BENCHMARK_SHARDS"

case "$SMU3M_BENCHMARK_PROFILE" in
  primary_fast)
    DEFAULT_BENCH_CPUS="1"
    DEFAULT_BENCH_MEM="8G"
    DEFAULT_BENCH_TIME="01:00:00"
    ;;
  scaffold)
    DEFAULT_BENCH_CPUS="1"
    DEFAULT_BENCH_MEM="8G"
    DEFAULT_BENCH_TIME="01:00:00"
    ;;
  full)
    DEFAULT_BENCH_CPUS="1"
    DEFAULT_BENCH_MEM="16G"
    DEFAULT_BENCH_TIME="04:00:00"
    ;;
  *)
    echo "ERROR: unsupported SMU3M_BENCHMARK_PROFILE=$SMU3M_BENCHMARK_PROFILE" >&2
    exit 2
    ;;
esac

BENCH_ACCOUNT="${SMU3M_BENCHMARK_SLURM_ACCOUNT:-${SMU3M_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
BENCH_TIME="${SMU3M_BENCHMARK_SLURM_TIME:-${SMU3M_SLURM_TIME:-$DEFAULT_BENCH_TIME}}"
BENCH_MEM="${SMU3M_BENCHMARK_SLURM_MEM:-${SMU3M_SLURM_MEM:-$DEFAULT_BENCH_MEM}}"
BENCH_CPUS="${SMU3M_BENCHMARK_SLURM_CPUS:-${SMU3M_SLURM_CPUS:-$DEFAULT_BENCH_CPUS}}"
BENCH_JOB_NAME="${SMU3M_BENCHMARK_SLURM_JOB_NAME:-smu3m-diff-bench}"
BENCH_LOG_DIR="${SMU3M_LOG_DIR:-$PROJECT_DIR/logs}"
BENCH_PARTITION="${SMU3M_BENCHMARK_SLURM_PARTITION:-${SMU3M_SLURM_PARTITION:-}}"
BENCH_ARRAY_CONCURRENCY="${SMU3M_BENCHMARK_ARRAY_CONCURRENCY:-$SMU3M_BENCHMARK_SHARDS}"

if (( SMU3M_BENCHMARK_SHARDS <= 0 )); then
  echo "ERROR: SMU3M_BENCHMARK_SHARDS must be positive, got $SMU3M_BENCHMARK_SHARDS" >&2
  exit 2
fi
case "$SMU3M_BENCHMARK_SUBMIT_MODE" in
  jobs | array) ;;
  *)
    echo "ERROR: unsupported SMU3M_BENCHMARK_SUBMIT_MODE=$SMU3M_BENCHMARK_SUBMIT_MODE" >&2
    echo "Use jobs or array." >&2
    exit 2
    ;;
esac

if [[ ! -x "$SMU3M_PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $SMU3M_PYTHON_BIN" >&2
  exit 2
fi
GENERATED_LATENTS="${SMU3M_GENERATED_LATENTS:-$SMU3M_OUTPUT_DIR/eval_latent/generated_latents.npy}"
if [[ ! -f "$GENERATED_LATENTS" ]]; then
  echo "ERROR: missing eval latents: $GENERATED_LATENTS" >&2
  echo "Run submit_unified_diffusion_refine.sh first." >&2
  exit 2
fi

mkdir -p "$BENCH_LOG_DIR"

echo "Submitting Unified 3M materialized benchmark"
echo "  output_dir=$SMU3M_OUTPUT_DIR"
echo "  multiproperty_output_dir=$SMMED_OUTPUT_DIR"
echo "  generated_latents=$GENERATED_LATENTS"
echo "  benchmark_profile=$SMU3M_BENCHMARK_PROFILE"
echo "  benchmark_shards=$SMU3M_BENCHMARK_SHARDS"
echo "  submit_mode=$SMU3M_BENCHMARK_SUBMIT_MODE"
echo "  source_similarity_rerank_candidates=$SMU3M_SOURCE_SIMILARITY_RERANK_CANDIDATES"
echo "  max_eval_per_property_count=$SMMED_MAX_EVAL_PER_PROPERTY_COUNT"
echo "  python=$SMU3M_PYTHON_BIN"
echo "  bench_cpus=$BENCH_CPUS"
echo "  bench_mem=$BENCH_MEM"
echo "  bench_time=$BENCH_TIME"
echo "  resource_note=serial benchmark defaults; override SMU3M_BENCHMARK_SLURM_CPUS/MEM/TIME for larger shards"

BENCH_SBATCH_ARGS=(
  --account="$BENCH_ACCOUNT"
  --job-name="$BENCH_JOB_NAME"
  --time="$BENCH_TIME"
  --mem="$BENCH_MEM"
  --cpus-per-task="$BENCH_CPUS"
  --export=ALL
)
if [[ -n "$BENCH_PARTITION" ]]; then
  BENCH_SBATCH_ARGS+=(--partition="$BENCH_PARTITION")
fi
bench_job_ids=()
if (( SMU3M_BENCHMARK_SHARDS > 1 )) && [[ "$SMU3M_BENCHMARK_SUBMIT_MODE" == "array" ]]; then
  BENCH_SBATCH_ARGS+=(--array="0-$((SMU3M_BENCHMARK_SHARDS - 1))%$BENCH_ARRAY_CONCURRENCY")
  BENCH_SBATCH_ARGS+=(--output="$BENCH_LOG_DIR/%x-%A_%a.log")
  bench_output="$(
    sbatch "${BENCH_SBATCH_ARGS[@]}" \
      --wrap="bash '$PROJECT_DIR/scripts/run_unified_materialized_benchmark.sh'"
  )"
  echo "$bench_output"
  bench_job_id="$(echo "$bench_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$bench_job_id" ]]; then
    echo "ERROR: failed to parse benchmark array job id." >&2
    exit 1
  fi
  bench_job_ids=("$bench_job_id")
elif (( SMU3M_BENCHMARK_SHARDS > 1 )); then
  for shard_index in $(seq 0 "$((SMU3M_BENCHMARK_SHARDS - 1))"); do
    shard_output="$(
      sbatch "${BENCH_SBATCH_ARGS[@]}" \
        --output="$BENCH_LOG_DIR/${BENCH_JOB_NAME}-s${shard_index}-%j.log" \
        --wrap="SMMED_EVAL_SHARD_INDEX=$shard_index bash '$PROJECT_DIR/scripts/run_unified_materialized_benchmark.sh'"
    )"
    echo "$shard_output"
    shard_job_id="$(echo "$shard_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ -z "$shard_job_id" ]]; then
      echo "ERROR: failed to parse benchmark shard job id for shard $shard_index." >&2
      exit 1
    fi
    bench_job_ids+=("$shard_job_id")
  done
else
  BENCH_SBATCH_ARGS+=(--output="$BENCH_LOG_DIR/%x-%j.log")
  bench_output="$(
    sbatch "${BENCH_SBATCH_ARGS[@]}" \
      --wrap="bash '$PROJECT_DIR/scripts/run_unified_materialized_benchmark.sh'"
  )"
  echo "$bench_output"
  bench_job_id="$(echo "$bench_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$bench_job_id" ]]; then
    echo "ERROR: failed to parse benchmark job id." >&2
    exit 1
  fi
  bench_job_ids=("$bench_job_id")
fi

case "$SMU3M_BENCHMARK_PROFILE" in
  full) default_output_dir="$SMU3M_OUTPUT_DIR/benchmark_materialized" ;;
  *) default_output_dir="$SMU3M_OUTPUT_DIR/benchmark_materialized_${SMU3M_BENCHMARK_PROFILE}" ;;
esac
if [[ -n "${SMU3M_BENCHMARK_OUTPUT_DIR:-}" ]]; then
  output_base_dir="$SMU3M_BENCHMARK_OUTPUT_DIR"
else
  output_base_dir="$default_output_dir"
fi
report_path="$output_base_dir/benchmark_report.md"

echo
echo "Materialized benchmark submitted."
echo "  job_ids=${bench_job_ids[*]}"
if (( SMU3M_BENCHMARK_SHARDS > 1 )); then
  merge_next="SMU3M_OUTPUT_DIR=$SMU3M_OUTPUT_DIR SMU3M_BENCHMARK_PROFILE=$SMU3M_BENCHMARK_PROFILE SMU3M_PYTHON_BIN=$SMU3M_PYTHON_BIN"
  if [[ -n "${SMU3M_BENCHMARK_OUTPUT_DIR:-}" ]]; then
    merge_next="$merge_next SMU3M_BENCHMARK_OUTPUT_DIR=$SMU3M_BENCHMARK_OUTPUT_DIR"
  fi
  merge_next="$merge_next bash $PROJECT_DIR/scripts/merge_unified_materialized_benchmark_shards.sh"
  if [[ "$SMU3M_BENCHMARK_SUBMIT_MODE" == "array" ]]; then
    echo "  logs=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-${bench_job_ids[0]}_<shard>.log"
  else
    echo "  logs=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-s<shard>-<job_id>.log"
  fi
  echo "  shard_reports=$output_base_dir/shards/shard_<shard>_of_${SMU3M_BENCHMARK_SHARDS}/benchmark_report.md"
  echo "  merge_next=$merge_next"
else
  echo "  log=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-${bench_job_ids[0]}.log"
  echo "  benchmark_report=$report_path"
fi
