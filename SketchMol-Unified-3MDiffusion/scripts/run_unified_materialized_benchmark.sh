#!/usr/bin/env bash
# Materialize Unified 3M latent eval outputs through the multi-property benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATASET_PROJECT_DIR="$REPO_DIR/SketchMol-MultiProperty-EditDataset"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SMU3M_PYTHON_BIN:-${SMMED_PYTHON_BIN:-${PYTHON_BIN:-python}}}"
UNIFIED_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v1}"
EVAL_LATENT_DIR="${SMU3M_EVAL_LATENT_DIR:-$UNIFIED_OUTPUT_DIR/eval_latent}"
UNIFIED_EVAL_JSONL="${SMU3M_EVAL_JSONL:-$UNIFIED_OUTPUT_DIR/dataset/unified_condition_eval.jsonl}"
GENERATED_LATENTS="${SMU3M_GENERATED_LATENTS:-$EVAL_LATENT_DIR/generated_latents.npy}"
EVAL_METRICS="${SMU3M_EVAL_METRICS:-$EVAL_LATENT_DIR/metrics.json}"
MULTIPROPERTY_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
CONDITION_ROWS="${SMMED_CONDITION_ROWS:-$MULTIPROPERTY_OUTPUT_DIR/condition_rows.csv}"
MOLECULE_DB="${SMMED_MOLECULE_DB_CSV:-$MULTIPROPERTY_OUTPUT_DIR/molecule_database.csv}"
BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
case "$BENCHMARK_PROFILE" in
  primary_fast)
    DEFAULT_METHODS="source_identity,scaffold_property_retrieval,edit_latent_source_similarity_rerank,edit_latent_scaffold_source_rerank,target_oracle"
    DEFAULT_MAX_EDIT_LATENT_CANDIDATES="5000"
    DEFAULT_SOURCE_SIMILARITY_RERANK_CANDIDATES="256"
    DEFAULT_BENCHMARK_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/benchmark_materialized_primary_fast"
    ;;
  scaffold)
    DEFAULT_METHODS="source_identity,scaffold_property_retrieval,edit_latent_scaffold_retrieval,edit_latent_scaffold_source_rerank,target_oracle"
    DEFAULT_MAX_EDIT_LATENT_CANDIDATES="20000"
    DEFAULT_SOURCE_SIMILARITY_RERANK_CANDIDATES="256"
    DEFAULT_BENCHMARK_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/benchmark_materialized_scaffold"
    ;;
  full)
    DEFAULT_METHODS="source_identity,scaffold_property_retrieval,edit_latent_global_retrieval,edit_latent_source_similarity_rerank,edit_latent_scaffold_retrieval,edit_latent_scaffold_source_rerank,target_oracle"
    DEFAULT_MAX_EDIT_LATENT_CANDIDATES="20000"
    DEFAULT_SOURCE_SIMILARITY_RERANK_CANDIDATES="512"
    DEFAULT_BENCHMARK_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/benchmark_materialized"
    ;;
  *)
    echo "ERROR: unsupported SMU3M_BENCHMARK_PROFILE=$BENCHMARK_PROFILE" >&2
    exit 2
    ;;
esac
BENCHMARK_OUTPUT_DIR="${SMU3M_BENCHMARK_OUTPUT_DIR:-$DEFAULT_BENCHMARK_OUTPUT_DIR}"
METHODS="${SMU3M_BENCHMARK_METHODS:-$DEFAULT_METHODS}"
FINGERPRINT_WEIGHT="${SMU3M_BENCHMARK_FINGERPRINT_WEIGHT:-1.0}"
PROPERTY_WEIGHT="${SMU3M_BENCHMARK_PROPERTY_WEIGHT:-1.0}"
DELTA_WEIGHT="${SMU3M_BENCHMARK_DELTA_WEIGHT:-0.35}"
DIRECTION_WEIGHT="${SMU3M_BENCHMARK_DIRECTION_WEIGHT:-0.10}"
SOURCE_SIMILARITY_WEIGHT="${SMU3M_BENCHMARK_SOURCE_SIMILARITY_WEIGHT:-1.0}"
SOURCE_FIRST_MIN_TANIMOTO="${SMU3M_SOURCE_FIRST_MIN_TANIMOTO:-0.4}"
SOURCE_FIRST_CANDIDATES="${SMU3M_SOURCE_FIRST_CANDIDATES:-0}"
SOURCE_TANIMOTO_THRESHOLDS="${SMMED_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"
MAX_GLOBAL_CANDIDATES="${SMMED_MAX_GLOBAL_CANDIDATES:-20000}"
MAX_EDIT_LATENT_CANDIDATES="${SMU3M_MAX_EDIT_LATENT_CANDIDATES:-$DEFAULT_MAX_EDIT_LATENT_CANDIDATES}"
SOURCE_SIMILARITY_RERANK_CANDIDATES="${SMU3M_SOURCE_SIMILARITY_RERANK_CANDIDATES:-$DEFAULT_SOURCE_SIMILARITY_RERANK_CANDIDATES}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"
RESTRICT_TO_EDIT_LATENT_INDEX="${SMU3M_RESTRICT_BENCHMARK_TO_EDIT_LATENT_INDEX:-1}"
SCAFFOLD_FALLBACK_MODE="${SMMED_SCAFFOLD_FALLBACK_MODE:-source_identity}"
LIMIT_EVAL_ROWS="${SMMED_LIMIT_EVAL_ROWS:-}"
EVAL_SHARD_COUNT="${SMMED_EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${SMMED_EVAL_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
EVAL_PREDICTIONS="${SMU3M_EVAL_PREDICTIONS:-$EVAL_LATENT_DIR/predictions.csv}"
EVAL_EXPORT_LIMIT="${SMU3M_EVAL_EXPORT_LIMIT:-}"
ALLOW_EVAL_TARGET_CANDIDATES="${SMMED_ALLOW_EVAL_TARGET_CANDIDATES:-0}"
SEED="${SMMED_SEED:-7}"

if (( EVAL_SHARD_COUNT > 1 )); then
  if [[ -n "${SMU3M_BENCHMARK_SHARD_OUTPUT_DIR:-}" ]]; then
    BENCHMARK_OUTPUT_DIR="$SMU3M_BENCHMARK_SHARD_OUTPUT_DIR"
  else
    BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR/shards/shard_${EVAL_SHARD_INDEX}_of_${EVAL_SHARD_COUNT}"
  fi
fi

export PYTHONPATH="$PROJECT_DIR:$DATASET_PROJECT_DIR:$REPO_DIR/SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"

echo "Unified 3M materialized benchmark"
echo "  python=$PYTHON_BIN"
echo "  unified_output_dir=$UNIFIED_OUTPUT_DIR"
echo "  eval_latent_dir=$EVAL_LATENT_DIR"
echo "  condition_rows=$CONDITION_ROWS"
echo "  molecule_db=$MOLECULE_DB"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  benchmark_profile=$BENCHMARK_PROFILE"
echo "  methods=$METHODS"
echo "  fingerprint_weight=$FINGERPRINT_WEIGHT"
echo "  max_edit_latent_candidates=$MAX_EDIT_LATENT_CANDIDATES"
echo "  source_similarity_rerank_candidates=$SOURCE_SIMILARITY_RERANK_CANDIDATES"
echo "  source_first_min_tanimoto=$SOURCE_FIRST_MIN_TANIMOTO"
echo "  source_first_candidates=$SOURCE_FIRST_CANDIDATES"
echo "  source_tanimoto_thresholds=$SOURCE_TANIMOTO_THRESHOLDS"
echo "  restrict_to_edit_latent_index=$RESTRICT_TO_EDIT_LATENT_INDEX"
echo "  eval_shard_index=$EVAL_SHARD_INDEX"
echo "  eval_shard_count=$EVAL_SHARD_COUNT"

for required in "$UNIFIED_EVAL_JSONL" "$GENERATED_LATENTS" "$CONDITION_ROWS" "$MOLECULE_DB"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required file not found: $required" >&2
    exit 2
  fi
done

if [[ ! -f "$EVAL_LATENT_DIR/edit_latent_predictions.npy" \
  || ! -f "$EVAL_LATENT_DIR/edit_latent_fingerprints.npy" \
  || ! -f "$EVAL_LATENT_DIR/index.csv" ]]; then
  mkdir -p "$EVAL_LATENT_DIR"
  EXPORT_ARGS=(
    --eval-jsonl "$UNIFIED_EVAL_JSONL"
    --latents-npy "$GENERATED_LATENTS"
    --output-dir "$EVAL_LATENT_DIR"
  )
  if [[ -f "$EVAL_METRICS" ]]; then
    EXPORT_ARGS+=(--metrics-json "$EVAL_METRICS")
  fi
  if [[ -f "$EVAL_PREDICTIONS" ]]; then
    EXPORT_ARGS+=(--predictions-csv "$EVAL_PREDICTIONS")
  fi
  if [[ -n "$EVAL_EXPORT_LIMIT" ]]; then
    EXPORT_ARGS+=(--limit "$EVAL_EXPORT_LIMIT")
  fi
  EXPORT_LOCK_DIR="$EVAL_LATENT_DIR/.benchmark_export.lock"
  if mkdir "$EXPORT_LOCK_DIR" 2>/dev/null; then
    cleanup_export_lock() {
      rmdir "$EXPORT_LOCK_DIR" 2>/dev/null || true
    }
    trap cleanup_export_lock EXIT
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_latent_benchmark_inputs.py" "${EXPORT_ARGS[@]}"
    cleanup_export_lock
    trap - EXIT
  else
    echo "Waiting for benchmark export lock: $EXPORT_LOCK_DIR"
    for _ in $(seq 1 720); do
      if [[ -f "$EVAL_LATENT_DIR/edit_latent_predictions.npy" \
        && -f "$EVAL_LATENT_DIR/edit_latent_fingerprints.npy" \
        && -f "$EVAL_LATENT_DIR/index.csv" ]]; then
        break
      fi
      sleep 10
    done
  fi
  if [[ ! -f "$EVAL_LATENT_DIR/edit_latent_predictions.npy" \
    || ! -f "$EVAL_LATENT_DIR/edit_latent_fingerprints.npy" \
    || ! -f "$EVAL_LATENT_DIR/index.csv" ]]; then
    echo "ERROR: timed out waiting for benchmark export files in $EVAL_LATENT_DIR" >&2
    exit 2
  fi
fi

BENCHMARK_ARGS=(
  --condition-rows-csv "$CONDITION_ROWS"
  --output-dir "$BENCHMARK_OUTPUT_DIR"
  --candidate-molecule-db-csv "$MOLECULE_DB"
  --methods "$METHODS"
  --edit-latent-dir "$EVAL_LATENT_DIR"
  --edit-latent-property-weight "$PROPERTY_WEIGHT"
  --edit-latent-delta-weight "$DELTA_WEIGHT"
  --edit-latent-direction-weight "$DIRECTION_WEIGHT"
  --edit-latent-fingerprint-weight "$FINGERPRINT_WEIGHT"
  --edit-latent-source-similarity-weight "$SOURCE_SIMILARITY_WEIGHT"
  --edit-latent-source-similarity-rerank-candidates "$SOURCE_SIMILARITY_RERANK_CANDIDATES"
  --source-first-min-tanimoto "$SOURCE_FIRST_MIN_TANIMOTO"
  --source-first-candidates "$SOURCE_FIRST_CANDIDATES"
  --max-global-candidates "$MAX_GLOBAL_CANDIDATES"
  --max-edit-latent-candidates "$MAX_EDIT_LATENT_CANDIDATES"
  --max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT"
  --eval-shard-count "$EVAL_SHARD_COUNT"
  --eval-shard-index "$EVAL_SHARD_INDEX"
  --scaffold-fallback-mode "$SCAFFOLD_FALLBACK_MODE"
  --source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS"
  --compute-tanimoto
  --seed "$SEED"
)
if [[ "$RESTRICT_TO_EDIT_LATENT_INDEX" == "1" ]]; then
  BENCHMARK_ARGS+=(--restrict-eval-to-edit-latent-index)
fi
if [[ -n "$LIMIT_EVAL_ROWS" ]]; then
  BENCHMARK_ARGS+=(--limit-eval-rows "$LIMIT_EVAL_ROWS")
fi
if [[ "$ALLOW_EVAL_TARGET_CANDIDATES" == "1" ]]; then
  BENCHMARK_ARGS+=(--allow-eval-target-candidates)
fi

"$PYTHON_BIN" "$DATASET_PROJECT_DIR/scripts/benchmark_multiproperty_retrieval.py" "${BENCHMARK_ARGS[@]}"

echo
echo "Unified materialized benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo
sed -n '1,90p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
