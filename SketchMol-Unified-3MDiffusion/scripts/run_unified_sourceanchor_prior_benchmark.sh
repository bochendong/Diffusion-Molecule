#!/usr/bin/env bash
# Run generated + prior-only materialized benchmarks for a source-anchor output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

SOURCEANCHOR_OUTPUT_ROOT="${SMU3M_SOURCEANCHOR_SWEEP_OUTPUT_ROOT:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceanchor_sweep_v1}"
SOURCEANCHOR_LABEL="${SMU3M_SOURCEANCHOR_BENCHMARK_LABEL:-blend095_guard050_p005}"
BASE_OUTPUT_DIR="${SMU3M_BASE_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_sourceguard_v1}"
OUTPUT_DIR="${SMU3M_SOURCEANCHOR_BENCHMARK_OUTPUT_DIR:-$SOURCEANCHOR_OUTPUT_ROOT/$SOURCEANCHOR_LABEL}"
BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
BENCHMARK_MODES="${SMU3M_SOURCEANCHOR_BENCHMARK_MODES:-generated,prior}"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
EVAL_JSONL="${SMU3M_EVAL_JSONL:-$OUTPUT_DIR/dataset/unified_condition_eval.jsonl}"
if [[ ! -f "$EVAL_JSONL" ]]; then
  EVAL_JSONL="$BASE_OUTPUT_DIR/dataset/unified_condition_eval.jsonl"
fi
if [[ ! -f "$EVAL_JSONL" ]]; then
  echo "ERROR: unified eval JSONL not found under $OUTPUT_DIR/dataset or $BASE_OUTPUT_DIR/dataset" >&2
  exit 2
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: source-anchor output dir not found: $OUTPUT_DIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: SMU3M_PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

run_mode() {
  local mode="$1"
  local generated_latents eval_latent_dir benchmark_output_dir job_label

  case "$mode" in
    generated)
      generated_latents="$OUTPUT_DIR/eval_latent/generated_latents.npy"
      eval_latent_dir="$OUTPUT_DIR/eval_latent"
      benchmark_output_dir="$OUTPUT_DIR/benchmark_materialized_${BENCHMARK_PROFILE}"
      job_label="generated"
      ;;
    prior | prior_only)
      generated_latents="$OUTPUT_DIR/eval_latent/prior_latents.npy"
      eval_latent_dir="$OUTPUT_DIR/eval_latent_prior_only"
      benchmark_output_dir="$OUTPUT_DIR/benchmark_materialized_prior_only_${BENCHMARK_PROFILE}"
      job_label="prior-only"
      ;;
    *)
      echo "ERROR: unsupported source-anchor benchmark mode: $mode" >&2
      exit 2
      ;;
  esac

  if [[ ! -f "$generated_latents" ]]; then
    echo "ERROR: missing $job_label latents: $generated_latents" >&2
    exit 2
  fi

  echo
  echo "Running source-anchor $job_label materialized benchmark"
  echo "  label=$SOURCEANCHOR_LABEL"
  echo "  output_dir=$OUTPUT_DIR"
  echo "  generated_latents=$generated_latents"
  echo "  benchmark_output_dir=$benchmark_output_dir"

  SMU3M_OUTPUT_DIR="$OUTPUT_DIR" \
  SMU3M_EVAL_JSONL="$EVAL_JSONL" \
  SMU3M_EVAL_LATENT_DIR="$eval_latent_dir" \
  SMU3M_GENERATED_LATENTS="$generated_latents" \
  SMU3M_EVAL_METRICS="$OUTPUT_DIR/eval_latent/metrics.json" \
  SMU3M_EVAL_PREDICTIONS="$OUTPUT_DIR/eval_latent/predictions.csv" \
  SMU3M_BENCHMARK_OUTPUT_DIR="$benchmark_output_dir" \
  SMU3M_BENCHMARK_PROFILE="$BENCHMARK_PROFILE" \
  SMU3M_BENCHMARK_SHARDS="${SMU3M_BENCHMARK_SHARDS:-${SMMED_EVAL_SHARD_COUNT:-1}}" \
  SMU3M_PYTHON_BIN="$PYTHON_BIN" \
  SMMED_EVAL_SHARD_COUNT="${SMMED_EVAL_SHARD_COUNT:-${SMU3M_BENCHMARK_SHARDS:-1}}" \
  SMMED_EVAL_SHARD_INDEX="${SMMED_EVAL_SHARD_INDEX:-0}" \
    bash "$PROJECT_DIR/scripts/run_unified_materialized_benchmark.sh"
}

IFS=',' read -r -a modes <<< "$BENCHMARK_MODES"
if (( ${#modes[@]} == 0 )); then
  echo "ERROR: no benchmark modes requested." >&2
  exit 2
fi

echo "Unified 3M source-anchor materialized benchmark"
echo "  label=$SOURCEANCHOR_LABEL"
echo "  output_dir=$OUTPUT_DIR"
echo "  benchmark_profile=$BENCHMARK_PROFILE"
echo "  benchmark_modes=$BENCHMARK_MODES"
echo "  eval_jsonl=$EVAL_JSONL"
echo "  python=$PYTHON_BIN"

for raw_mode in "${modes[@]}"; do
  mode="$(printf '%s' "$raw_mode" | xargs)"
  [[ -z "$mode" ]] && continue
  run_mode "$mode"
done

echo
echo "Source-anchor materialized benchmarks finished."
