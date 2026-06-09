#!/usr/bin/env bash
# Submit materialized benchmarks for the source-aware sweep winners.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

SWEEP_OUTPUT_ROOT="${SMU3M_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2}"
WINNER_LABELS="${SMU3M_WINNER_LABELS:-hard002_head,balanced_005_001}"
BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
BENCHMARK_SHARDS="${SMU3M_BENCHMARK_SHARDS:-5}"
BENCHMARK_SUBMIT_MODE="${SMU3M_BENCHMARK_SUBMIT_MODE:-array}"
BENCHMARK_PRIOR_ONLY="${SMU3M_BENCHMARK_PRIOR_ONLY:-1}"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
JOB_PREFIX="${SMU3M_WINNER_BENCHMARK_JOB_PREFIX:-smu3m-srcwin}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

echo "Submitting source-aware winner materialized benchmarks"
echo "  sweep_output_root=$SWEEP_OUTPUT_ROOT"
echo "  winner_labels=$WINNER_LABELS"
echo "  benchmark_profile=$BENCHMARK_PROFILE"
echo "  benchmark_shards=$BENCHMARK_SHARDS"
echo "  benchmark_submit_mode=$BENCHMARK_SUBMIT_MODE"
echo "  benchmark_prior_only=$BENCHMARK_PRIOR_ONLY"
echo "  python=$PYTHON_BIN"

resolve_output_spec() {
  local spec="$1"
  if [[ "$spec" == *"="* ]]; then
    RESOLVED_LABEL="${spec%%=*}"
    RESOLVED_OUTPUT_DIR="${spec#*=}"
  elif [[ "$spec" == */* ]]; then
    RESOLVED_OUTPUT_DIR="$spec"
    RESOLVED_LABEL="$(basename "$spec")"
  else
    RESOLVED_LABEL="$spec"
    RESOLVED_OUTPUT_DIR="$SWEEP_OUTPUT_ROOT/$spec"
  fi
}

submit_benchmark() {
  local spec="$1"
  local mode="$2"
  local label output_dir
  resolve_output_spec "$spec"
  label="$RESOLVED_LABEL"
  output_dir="$RESOLVED_OUTPUT_DIR"
  local safe_label
  safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9_.-' '_')"

  if [[ ! -d "$output_dir" ]]; then
    echo "ERROR: benchmark output dir not found for $label: $output_dir" >&2
    exit 2
  fi

  local benchmark_output_dir job_name
  local generated_latents eval_latent_dir eval_metrics eval_predictions
  if [[ "$mode" == "generated" ]]; then
    generated_latents="$output_dir/eval_latent/generated_latents.npy"
    eval_latent_dir="$output_dir/eval_latent"
    eval_metrics="$output_dir/eval_latent/metrics.json"
    eval_predictions="$output_dir/eval_latent/predictions.csv"
    benchmark_output_dir="$output_dir/benchmark_materialized_${BENCHMARK_PROFILE}"
    job_name="${JOB_PREFIX}-${safe_label}-gen"
  elif [[ "$mode" == "prior" ]]; then
    generated_latents="$output_dir/eval_latent/prior_latents.npy"
    eval_latent_dir="$output_dir/eval_latent_prior_only"
    eval_metrics="$output_dir/eval_latent/metrics.json"
    eval_predictions="$output_dir/eval_latent/predictions.csv"
    benchmark_output_dir="$output_dir/benchmark_materialized_prior_only_${BENCHMARK_PROFILE}"
    job_name="${JOB_PREFIX}-${safe_label}-prior"
  else
    echo "ERROR: unsupported benchmark mode=$mode" >&2
    exit 2
  fi

  if [[ ! -f "$generated_latents" ]]; then
    echo "ERROR: missing $mode latents for $label: $generated_latents" >&2
    exit 2
  fi

  echo
  echo "Submitting $mode benchmark for $label"
  echo "  output_dir=$output_dir"
  SMU3M_OUTPUT_DIR="$output_dir" \
  SMU3M_EVAL_LATENT_DIR="$eval_latent_dir" \
  SMU3M_GENERATED_LATENTS="$generated_latents" \
  SMU3M_EVAL_METRICS="$eval_metrics" \
  SMU3M_EVAL_PREDICTIONS="$eval_predictions" \
  SMU3M_BENCHMARK_OUTPUT_DIR="$benchmark_output_dir" \
  SMU3M_BENCHMARK_PROFILE="$BENCHMARK_PROFILE" \
  SMU3M_BENCHMARK_SHARDS="$BENCHMARK_SHARDS" \
  SMU3M_BENCHMARK_SUBMIT_MODE="$BENCHMARK_SUBMIT_MODE" \
  SMU3M_BENCHMARK_SLURM_JOB_NAME="$job_name" \
  SMU3M_PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_DIR/scripts/submit_unified_materialized_benchmark.sh"
}

IFS=',' read -r -a labels <<< "$WINNER_LABELS"
if (( ${#labels[@]} == 0 )); then
  echo "ERROR: no winner labels provided." >&2
  exit 2
fi

for raw_label in "${labels[@]}"; do
  label="$(printf '%s' "$raw_label" | xargs)"
  [[ -z "$label" ]] && continue
  submit_benchmark "$label" generated
  if [[ "$BENCHMARK_PRIOR_ONLY" == "1" ]]; then
    submit_benchmark "$label" prior
  fi
done

echo
echo "Winner materialized benchmark submissions finished."
