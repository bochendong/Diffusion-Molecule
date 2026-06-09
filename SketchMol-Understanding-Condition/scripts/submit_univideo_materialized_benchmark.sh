#!/usr/bin/env bash
# Submit the UniVideo OCR-free, materialized benchmark as a separate Slurm job.

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

export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-$SUCC_DEFAULT_UNIVIDEO_OUTPUT_DIR}"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
export SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES="${SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES:-256}"

case "$SUCC_MATERIALIZED_BENCHMARK_PROFILE" in
  primary_fast | latent)
    DEFAULT_BENCH_CPUS="1"
    DEFAULT_BENCH_MEM="16G"
    DEFAULT_BENCH_TIME="02:00:00"
    ;;
  oracle)
    DEFAULT_BENCH_CPUS="1"
    DEFAULT_BENCH_MEM="8G"
    DEFAULT_BENCH_TIME="01:00:00"
    ;;
  *)
    echo "ERROR: unsupported SUCC_MATERIALIZED_BENCHMARK_PROFILE=$SUCC_MATERIALIZED_BENCHMARK_PROFILE" >&2
    echo "       Use primary_fast, oracle, or latent." >&2
    exit 2
    ;;
esac

BENCH_ACCOUNT="${SUCC_MATERIALIZED_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
BENCH_TIME="${SUCC_MATERIALIZED_SLURM_TIME:-${SUCC_SLURM_TIME:-$DEFAULT_BENCH_TIME}}"
BENCH_MEM="${SUCC_MATERIALIZED_SLURM_MEM:-${SUCC_SLURM_MEM:-$DEFAULT_BENCH_MEM}}"
BENCH_CPUS="${SUCC_MATERIALIZED_SLURM_CPUS:-${SUCC_SLURM_CPUS:-$DEFAULT_BENCH_CPUS}}"
BENCH_JOB_NAME="${SUCC_MATERIALIZED_SLURM_JOB_NAME:-succ-univideo-bench}"
BENCH_LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
BENCH_PARTITION="${SUCC_MATERIALIZED_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
BENCH_DEPENDENCY="${SUCC_MATERIALIZED_SLURM_DEPENDENCY:-}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

IMAGE_CSV="${SUCC_IMAGE_CSV:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/image_structure_benchmark/image_path.csv}"
GENERATED_LATENTS="${SUCC_GENERATED_LATENTS:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/generated_latents.npy}"
CANDIDATE_LATENTS="${SUCC_CANDIDATE_LATENTS:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/target_latents.npy}"
if [[ ! -f "$IMAGE_CSV" ]]; then
  echo "ERROR: missing image CSV: $IMAGE_CSV" >&2
  echo "Run the UniVideo pipeline with SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=prepare first." >&2
  exit 2
fi
if [[ "$SUCC_MATERIALIZED_BENCHMARK_PROFILE" != "oracle" ]]; then
  for required in "$GENERATED_LATENTS" "$CANDIDATE_LATENTS"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: missing latent file: $required" >&2
      echo "Use SUCC_MATERIALIZED_BENCHMARK_PROFILE=oracle if only image_path.csv is available." >&2
      exit 2
    fi
  done
fi

mkdir -p "$BENCH_LOG_DIR"

echo "Submitting UniVideo OCR-free materialized benchmark"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  image_csv=$IMAGE_CSV"
echo "  generated_latents=$GENERATED_LATENTS"
echo "  candidate_latents=$CANDIDATE_LATENTS"
echo "  benchmark_profile=$SUCC_MATERIALIZED_BENCHMARK_PROFILE"
echo "  source_similarity_rerank_candidates=$SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES"
echo "  python=$SUCC_PYTHON_BIN"
echo "  bench_cpus=$BENCH_CPUS"
echo "  bench_mem=$BENCH_MEM"
echo "  bench_time=$BENCH_TIME"
if [[ -n "$BENCH_DEPENDENCY" ]]; then
  echo "  dependency=$BENCH_DEPENDENCY"
fi

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
if [[ -n "$BENCH_DEPENDENCY" ]]; then
  BENCH_SBATCH_ARGS+=(--dependency="$BENCH_DEPENDENCY")
fi

bench_output="$(
  sbatch "${BENCH_SBATCH_ARGS[@]}" \
    --wrap="bash '$PROJECT_DIR/scripts/run_univideo_materialized_benchmark.sh'"
)"
echo "$bench_output"
bench_job_id="$(echo "$bench_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$bench_job_id" ]]; then
  echo "ERROR: failed to parse benchmark job id." >&2
  exit 1
fi

case "$SUCC_MATERIALIZED_BENCHMARK_PROFILE" in
  primary_fast) default_output_dir="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_primary_fast" ;;
  oracle) default_output_dir="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_oracle" ;;
  latent) default_output_dir="$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_latent" ;;
esac
output_base_dir="${SUCC_MATERIALIZED_BENCHMARK_OUTPUT_DIR:-$default_output_dir}"

echo
echo "UniVideo materialized benchmark submitted."
echo "  job_id=$bench_job_id"
echo "  log=$BENCH_LOG_DIR/${BENCH_JOB_NAME}-${bench_job_id}.log"
echo "  benchmark_report=$output_base_dir/benchmark_report.md"
