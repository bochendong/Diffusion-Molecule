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
BENCHMARK_OUTPUT_DIR="${SMU3M_BENCHMARK_OUTPUT_DIR:-$UNIFIED_OUTPUT_DIR/benchmark_materialized}"
METHODS="${SMU3M_BENCHMARK_METHODS:-source_identity,scaffold_property_retrieval,edit_latent_global_retrieval,edit_latent_source_similarity_rerank,edit_latent_scaffold_retrieval,edit_latent_scaffold_source_rerank,target_oracle}"
FINGERPRINT_WEIGHT="${SMU3M_BENCHMARK_FINGERPRINT_WEIGHT:-1.0}"
PROPERTY_WEIGHT="${SMU3M_BENCHMARK_PROPERTY_WEIGHT:-1.0}"
DELTA_WEIGHT="${SMU3M_BENCHMARK_DELTA_WEIGHT:-0.35}"
DIRECTION_WEIGHT="${SMU3M_BENCHMARK_DIRECTION_WEIGHT:-0.10}"
SOURCE_SIMILARITY_WEIGHT="${SMU3M_BENCHMARK_SOURCE_SIMILARITY_WEIGHT:-1.0}"
SOURCE_TANIMOTO_THRESHOLDS="${SMMED_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"
MAX_GLOBAL_CANDIDATES="${SMMED_MAX_GLOBAL_CANDIDATES:-20000}"
MAX_EDIT_LATENT_CANDIDATES="${SMU3M_MAX_EDIT_LATENT_CANDIDATES:-20000}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"
RESTRICT_TO_EDIT_LATENT_INDEX="${SMU3M_RESTRICT_BENCHMARK_TO_EDIT_LATENT_INDEX:-1}"
SCAFFOLD_FALLBACK_MODE="${SMMED_SCAFFOLD_FALLBACK_MODE:-source_identity}"
LIMIT_EVAL_ROWS="${SMMED_LIMIT_EVAL_ROWS:-}"
EVAL_PREDICTIONS="${SMU3M_EVAL_PREDICTIONS:-$EVAL_LATENT_DIR/predictions.csv}"
EVAL_EXPORT_LIMIT="${SMU3M_EVAL_EXPORT_LIMIT:-}"
ALLOW_EVAL_TARGET_CANDIDATES="${SMMED_ALLOW_EVAL_TARGET_CANDIDATES:-0}"
SEED="${SMMED_SEED:-7}"

export PYTHONPATH="$PROJECT_DIR:$DATASET_PROJECT_DIR:$REPO_DIR/SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"

echo "Unified 3M materialized benchmark"
echo "  python=$PYTHON_BIN"
echo "  unified_output_dir=$UNIFIED_OUTPUT_DIR"
echo "  eval_latent_dir=$EVAL_LATENT_DIR"
echo "  condition_rows=$CONDITION_ROWS"
echo "  molecule_db=$MOLECULE_DB"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  methods=$METHODS"
echo "  fingerprint_weight=$FINGERPRINT_WEIGHT"
echo "  source_tanimoto_thresholds=$SOURCE_TANIMOTO_THRESHOLDS"
echo "  restrict_to_edit_latent_index=$RESTRICT_TO_EDIT_LATENT_INDEX"

for required in "$UNIFIED_EVAL_JSONL" "$GENERATED_LATENTS" "$CONDITION_ROWS" "$MOLECULE_DB"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required file not found: $required" >&2
    exit 2
  fi
done

if [[ ! -f "$EVAL_LATENT_DIR/edit_latent_predictions.npy" \
  || ! -f "$EVAL_LATENT_DIR/edit_latent_fingerprints.npy" \
  || ! -f "$EVAL_LATENT_DIR/index.csv" ]]; then
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
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_latent_benchmark_inputs.py" "${EXPORT_ARGS[@]}"
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT_EVAL_ROWS" ]]; then
  LIMIT_ARGS=(--limit-eval-rows "$LIMIT_EVAL_ROWS")
fi
TARGET_POOL_ARGS=()
if [[ "$ALLOW_EVAL_TARGET_CANDIDATES" == "1" ]]; then
  TARGET_POOL_ARGS=(--allow-eval-target-candidates)
fi
RESTRICT_ARGS=()
if [[ "$RESTRICT_TO_EDIT_LATENT_INDEX" == "1" ]]; then
  RESTRICT_ARGS=(--restrict-eval-to-edit-latent-index)
fi

"$PYTHON_BIN" "$DATASET_PROJECT_DIR/scripts/benchmark_multiproperty_retrieval.py" \
  --condition-rows-csv "$CONDITION_ROWS" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --candidate-molecule-db-csv "$MOLECULE_DB" \
  --methods "$METHODS" \
  --edit-latent-dir "$EVAL_LATENT_DIR" \
  "${RESTRICT_ARGS[@]}" \
  --edit-latent-property-weight "$PROPERTY_WEIGHT" \
  --edit-latent-delta-weight "$DELTA_WEIGHT" \
  --edit-latent-direction-weight "$DIRECTION_WEIGHT" \
  --edit-latent-fingerprint-weight "$FINGERPRINT_WEIGHT" \
  --edit-latent-source-similarity-weight "$SOURCE_SIMILARITY_WEIGHT" \
  --max-global-candidates "$MAX_GLOBAL_CANDIDATES" \
  --max-edit-latent-candidates "$MAX_EDIT_LATENT_CANDIDATES" \
  --max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT" \
  --scaffold-fallback-mode "$SCAFFOLD_FALLBACK_MODE" \
  --source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS" \
  --compute-tanimoto \
  --seed "$SEED" \
  "${LIMIT_ARGS[@]}" \
  "${TARGET_POOL_ARGS[@]}"

echo
echo "Unified materialized benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo
sed -n '1,90p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
