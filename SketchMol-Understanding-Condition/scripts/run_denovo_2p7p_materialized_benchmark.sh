#!/usr/bin/env bash
# Run an OCR-free de novo 2p-7p property-design benchmark aligned to SketchMol.

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
OUTPUT_DIR="${SUCC_DENOVO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/denovo_2p7p_v1}"
MOLECULE_DB="${SUCC_DENOVO_MOLECULE_DB_CSV:-$SMMED_DEFAULT_MOLECULE_DB}"
BENCHMARK_ROWS_CSV="${SUCC_DENOVO_BENCHMARK_ROWS_CSV:-$OUTPUT_DIR/denovo_2p7p_rows.csv}"
CANDIDATE_CSV="${SUCC_DENOVO_CANDIDATE_CSV:-$OUTPUT_DIR/denovo_candidate_rows.csv}"
ROWS_PER_PROPERTY_COUNT="${SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT:-1000}"
MIN_PROPERTIES="${SUCC_DENOVO_MIN_PROPERTIES:-2}"
MAX_PROPERTIES="${SUCC_DENOVO_MAX_PROPERTIES:-7}"
SEED="${SUCC_DENOVO_SEED:-13}"
CANDIDATE_LIMIT="${SUCC_DENOVO_CANDIDATE_LIMIT:-0}"
METHODS="${SUCC_DENOVO_MATERIALIZED_METHODS:-property_nearest,target_oracle}"
BENCHMARK_OUTPUT_DIR="${SUCC_DENOVO_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_materialized}"
DIRECT_CSV="${SUCC_DENOVO_TARGET_MOLECULES_DIRECT_CSV:-$BENCHMARK_OUTPUT_DIR/target_molecules_direct.csv}"
TOP_K="${SUCC_DENOVO_TARGET_FINDER_TOP_K:-5}"
SOURCE_TANIMOTO_THRESHOLDS="${SUCC_DENOVO_SOURCE_TANIMOTO_THRESHOLDS:-}"
FORCE_EXPORT="${SUCC_DENOVO_FORCE_EXPORT:-0}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

echo "De novo 2p-7p materialized benchmark"
echo "  python=$PYTHON_BIN"
echo "  molecule_db=$MOLECULE_DB"
echo "  output_dir=$OUTPUT_DIR"
echo "  benchmark_rows=$BENCHMARK_ROWS_CSV"
echo "  candidate_csv=$CANDIDATE_CSV"
echo "  rows_per_property_count=$ROWS_PER_PROPERTY_COUNT"
echo "  property_range=${MIN_PROPERTIES}-${MAX_PROPERTIES}"
echo "  methods=$METHODS"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"

if [[ ! -f "$MOLECULE_DB" ]]; then
  cat <<EOF >&2
ERROR: molecule database not found: $MOLECULE_DB

Build it first, for example:
  bash SketchMol-MultiProperty-EditDataset/scripts/run_full_benchmark.sh

Or point to an existing database:
  SUCC_DENOVO_MOLECULE_DB_CSV=/path/to/molecule_database.csv \\
  bash SketchMol-Understanding-Condition/scripts/run_denovo_2p7p_materialized_benchmark.sh
EOF
  exit 2
fi

if [[ "$FORCE_EXPORT" == "1" || ! -f "$BENCHMARK_ROWS_CSV" || ! -f "$CANDIDATE_CSV" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_denovo_2p7p_benchmark_rows.py" \
    --molecule-db-csv "$MOLECULE_DB" \
    --output-csv "$BENCHMARK_ROWS_CSV" \
    --candidate-output-csv "$CANDIDATE_CSV" \
    --rows-per-property-count "$ROWS_PER_PROPERTY_COUNT" \
    --min-properties "$MIN_PROPERTIES" \
    --max-properties "$MAX_PROPERTIES" \
    --seed "$SEED" \
    --candidate-limit "$CANDIDATE_LIMIT"
fi

mkdir -p "$BENCHMARK_OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/materialize_univideo_target_molecules.py" \
  --source-csv "$BENCHMARK_ROWS_CSV" \
  --candidate-csv "$CANDIDATE_CSV" \
  --output-csv "$DIRECT_CSV" \
  --methods "$METHODS" \
  --top-k "$TOP_K"

EVAL_ARGS=(
  --image-csv "$DIRECT_CSV"
  --output-dir "$BENCHMARK_OUTPUT_DIR"
  --method "denovo_2p7p_materialized"
  --smiles-column generated_smiles
  --report-title "De Novo 2p-7p Property-Design Benchmark"
  --benchmark-family "denovo_property_design"
  --benchmark-task "denovo_2p7p_property_design"
  --accept-direct-smiles
  --hide-source-similarity-section
)
if [[ -n "$SOURCE_TANIMOTO_THRESHOLDS" ]]; then
  EVAL_ARGS+=(--source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" "${EVAL_ARGS[@]}"

echo
echo "De novo 2p-7p benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo "  direct_csv=$DIRECT_CSV"
echo
sed -n '1,90p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
