#!/usr/bin/env bash
# Materialize UniVideo eval outputs through an OCR-free, Unified-3M-style benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-$SUCC_DEFAULT_UNIVIDEO_OUTPUT_DIR}"
EVAL_LATENT_DIR="${SUCC_EVAL_LATENT_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent}"
DATASET_DIR="${SUCC_UNIVIDEO_DATASET_DIR:-$UNIFIED_OUTPUT_DIR/dataset}"
EVAL_JSONL="${SUCC_EVAL_JSONL:-$DATASET_DIR/univideo_edit_eval.jsonl}"
PREDICTIONS_CSV="${SUCC_PREDICTIONS_CSV:-$EVAL_LATENT_DIR/predictions.csv}"
BENCHMARK_ROWS_CSV="${SUCC_BENCHMARK_ROWS_CSV:-$UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_condition_rows.csv}"
SOURCE_CSV="${SUCC_SOURCE_CSV:-${SUCC_IMAGE_CSV:-$BENCHMARK_ROWS_CSV}}"
CANDIDATE_CSV="${SUCC_TARGET_FINDER_CANDIDATE_CSV:-$SOURCE_CSV}"
GENERATED_LATENTS="${SUCC_GENERATED_LATENTS:-$EVAL_LATENT_DIR/generated_latents.npy}"
CANDIDATE_LATENTS="${SUCC_CANDIDATE_LATENTS:-$EVAL_LATENT_DIR/target_latents.npy}"

BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
case "$BENCHMARK_PROFILE" in
  primary_fast)
    DEFAULT_METHODS="source_identity,source_tanimoto_property_oracle,edit_latent_source_first_rerank,edit_latent_source_similarity_rerank,target_oracle"
    DEFAULT_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_primary_fast"
    DEFAULT_RERANK_CANDIDATES="256"
    ;;
  oracle)
    DEFAULT_METHODS="source_identity,source_tanimoto_property_oracle,target_oracle"
    DEFAULT_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_oracle"
    DEFAULT_RERANK_CANDIDATES="0"
    ;;
  latent)
    DEFAULT_METHODS="source_identity,edit_latent_source_similarity_rerank,target_oracle"
    DEFAULT_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_latent"
    DEFAULT_RERANK_CANDIDATES="256"
    ;;
  table_attack)
    DEFAULT_METHODS="source_identity,source_tanimoto_property_oracle,source_tanimoto_table_success_oracle,edit_latent_source_similarity_rerank,edit_latent_table_success_rerank,target_oracle"
    DEFAULT_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_table_attack"
    DEFAULT_RERANK_CANDIDATES="1024"
    ;;
  *)
    echo "ERROR: unsupported SUCC_MATERIALIZED_BENCHMARK_PROFILE=$BENCHMARK_PROFILE" >&2
    echo "       Use primary_fast, oracle, latent, or table_attack." >&2
    exit 2
    ;;
esac

METHODS="${SUCC_MATERIALIZED_METHODS:-$DEFAULT_METHODS}"
BENCHMARK_OUTPUT_DIR="${SUCC_MATERIALIZED_BENCHMARK_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
DIRECT_CSV="${SUCC_TARGET_MOLECULES_DIRECT_CSV:-$BENCHMARK_OUTPUT_DIR/target_molecules_direct.csv}"
SOURCE_FIRST_MIN_TANIMOTO="${SUCC_SOURCE_FIRST_MIN_TANIMOTO:-0.4}"
SOURCE_FIRST_CANDIDATES="${SUCC_SOURCE_FIRST_CANDIDATES:-0}"
SOURCE_SIMILARITY_WEIGHT="${SUCC_SOURCE_SIMILARITY_WEIGHT:-1.0}"
SOURCE_SIMILARITY_RERANK_CANDIDATES="${SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES:-$DEFAULT_RERANK_CANDIDATES}"
TABLE_SUCCESS_RERANK_CANDIDATES="${SUCC_TABLE_SUCCESS_RERANK_CANDIDATES:-$DEFAULT_RERANK_CANDIDATES}"
TABLE_SUCCESS_WEIGHT="${SUCC_TABLE_SUCCESS_WEIGHT:-100.0}"
TABLE_SOURCE_WEIGHT="${SUCC_TABLE_SOURCE_WEIGHT:-5.0}"
TABLE_LATENT_WEIGHT="${SUCC_TABLE_LATENT_WEIGHT:-1.0}"
TOP_K="${SUCC_TARGET_FINDER_TOP_K:-5}"
SOURCE_TANIMOTO_THRESHOLDS="${SUCC_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "UniVideo OCR-free materialized benchmark"
echo "  python=$PYTHON_BIN"
echo "  unified_output_dir=$UNIFIED_OUTPUT_DIR"
echo "  eval_latent_dir=$EVAL_LATENT_DIR"
echo "  eval_jsonl=$EVAL_JSONL"
echo "  predictions_csv=$PREDICTIONS_CSV"
echo "  source_csv=$SOURCE_CSV"
echo "  candidate_csv=$CANDIDATE_CSV"
echo "  generated_latents=$GENERATED_LATENTS"
echo "  candidate_latents=$CANDIDATE_LATENTS"
echo "  benchmark_profile=$BENCHMARK_PROFILE"
echo "  methods=$METHODS"
echo "  output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  direct_csv=$DIRECT_CSV"
echo "  source_first_min_tanimoto=$SOURCE_FIRST_MIN_TANIMOTO"
echo "  source_similarity_rerank_candidates=$SOURCE_SIMILARITY_RERANK_CANDIDATES"
echo "  table_success_rerank_candidates=$TABLE_SUCCESS_RERANK_CANDIDATES"
echo "  table_success_weight=$TABLE_SUCCESS_WEIGHT"
echo "  table_source_weight=$TABLE_SOURCE_WEIGHT"
echo "  source_tanimoto_thresholds=$SOURCE_TANIMOTO_THRESHOLDS"

if [[ ! -f "$SOURCE_CSV" ]]; then
  if [[ "$SOURCE_CSV" == "$BENCHMARK_ROWS_CSV" ]]; then
    for required in "$EVAL_JSONL" "$PREDICTIONS_CSV"; do
      if [[ ! -f "$required" ]]; then
        echo "ERROR: required file not found for benchmark row export: $required" >&2
        exit 2
      fi
    done
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_univideo_benchmark_rows.py" \
      --predictions-csv "$PREDICTIONS_CSV" \
      --eval-jsonl "$EVAL_JSONL" \
      --output-csv "$SOURCE_CSV" \
      --method "univideo_${BENCHMARK_PROFILE}"
  else
    echo "ERROR: source CSV not found: $SOURCE_CSV" >&2
    exit 2
  fi
fi
if [[ ! -f "$CANDIDATE_CSV" ]]; then
  echo "ERROR: candidate CSV not found: $CANDIDATE_CSV" >&2
  exit 2
fi
if [[ "$METHODS" == *"edit_latent"* ]]; then
  for required in "$GENERATED_LATENTS" "$CANDIDATE_LATENTS"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: required latent file not found: $required" >&2
      echo "Use SUCC_MATERIALIZED_BENCHMARK_PROFILE=oracle if only benchmark condition rows are available." >&2
      exit 2
    fi
  done
fi

mkdir -p "$BENCHMARK_OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/materialize_univideo_target_molecules.py" \
  --source-csv "$SOURCE_CSV" \
  --candidate-csv "$CANDIDATE_CSV" \
  --output-csv "$DIRECT_CSV" \
  --methods "$METHODS" \
  --generated-latents-npy "$GENERATED_LATENTS" \
  --candidate-latents-npy "$CANDIDATE_LATENTS" \
  --top-k "$TOP_K" \
  --source-first-min-tanimoto "$SOURCE_FIRST_MIN_TANIMOTO" \
  --source-first-candidates "$SOURCE_FIRST_CANDIDATES" \
  --source-similarity-weight "$SOURCE_SIMILARITY_WEIGHT" \
  --source-similarity-rerank-candidates "$SOURCE_SIMILARITY_RERANK_CANDIDATES" \
  --table-success-rerank-candidates "$TABLE_SUCCESS_RERANK_CANDIDATES" \
  --table-success-weight "$TABLE_SUCCESS_WEIGHT" \
  --table-source-weight "$TABLE_SOURCE_WEIGHT" \
  --table-latent-weight "$TABLE_LATENT_WEIGHT"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
  --image-csv "$DIRECT_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --method "univideo_materialized" \
  --smiles-column generated_smiles \
  --accept-direct-smiles \
  --source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS"

echo
echo "UniVideo materialized benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo "  direct_csv=$DIRECT_CSV"
echo
sed -n '1,90p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
