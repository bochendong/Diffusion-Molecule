#!/usr/bin/env bash
# Materialize UniVideo eval outputs through an OCR-free, Unified-3M-style benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1}"
EVAL_LATENT_DIR="${SUCC_EVAL_LATENT_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent}"
STRUCTURE_BENCHMARK_DIR="${SUCC_STRUCTURE_BENCHMARK_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/image_structure_benchmark}"
IMAGE_CSV="${SUCC_IMAGE_CSV:-$STRUCTURE_BENCHMARK_DIR/image_path.csv}"
CANDIDATE_CSV="${SUCC_TARGET_FINDER_CANDIDATE_CSV:-$IMAGE_CSV}"
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
  *)
    echo "ERROR: unsupported SUCC_MATERIALIZED_BENCHMARK_PROFILE=$BENCHMARK_PROFILE" >&2
    echo "       Use primary_fast, oracle, or latent." >&2
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
TOP_K="${SUCC_TARGET_FINDER_TOP_K:-5}"
SOURCE_TANIMOTO_THRESHOLDS="${SUCC_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "UniVideo OCR-free materialized benchmark"
echo "  python=$PYTHON_BIN"
echo "  unified_output_dir=$UNIFIED_OUTPUT_DIR"
echo "  eval_latent_dir=$EVAL_LATENT_DIR"
echo "  image_csv=$IMAGE_CSV"
echo "  candidate_csv=$CANDIDATE_CSV"
echo "  generated_latents=$GENERATED_LATENTS"
echo "  candidate_latents=$CANDIDATE_LATENTS"
echo "  benchmark_profile=$BENCHMARK_PROFILE"
echo "  methods=$METHODS"
echo "  output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  direct_csv=$DIRECT_CSV"
echo "  source_first_min_tanimoto=$SOURCE_FIRST_MIN_TANIMOTO"
echo "  source_similarity_rerank_candidates=$SOURCE_SIMILARITY_RERANK_CANDIDATES"
echo "  source_tanimoto_thresholds=$SOURCE_TANIMOTO_THRESHOLDS"

if [[ ! -f "$IMAGE_CSV" ]]; then
  echo "ERROR: image CSV not found: $IMAGE_CSV" >&2
  echo "Run the UniVideo pipeline with SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=prepare first." >&2
  exit 2
fi
if [[ ! -f "$CANDIDATE_CSV" ]]; then
  echo "ERROR: candidate CSV not found: $CANDIDATE_CSV" >&2
  exit 2
fi
if [[ "$METHODS" == *"edit_latent"* ]]; then
  for required in "$GENERATED_LATENTS" "$CANDIDATE_LATENTS"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: required latent file not found: $required" >&2
      echo "Use SUCC_MATERIALIZED_BENCHMARK_PROFILE=oracle if only image_path.csv is available." >&2
      exit 2
    fi
  done
fi

mkdir -p "$BENCHMARK_OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/materialize_univideo_target_molecules.py" \
  --source-csv "$IMAGE_CSV" \
  --candidate-csv "$CANDIDATE_CSV" \
  --output-csv "$DIRECT_CSV" \
  --methods "$METHODS" \
  --generated-latents-npy "$GENERATED_LATENTS" \
  --candidate-latents-npy "$CANDIDATE_LATENTS" \
  --top-k "$TOP_K" \
  --source-first-min-tanimoto "$SOURCE_FIRST_MIN_TANIMOTO" \
  --source-first-candidates "$SOURCE_FIRST_CANDIDATES" \
  --source-similarity-weight "$SOURCE_SIMILARITY_WEIGHT" \
  --source-similarity-rerank-candidates "$SOURCE_SIMILARITY_RERANK_CANDIDATES"

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
