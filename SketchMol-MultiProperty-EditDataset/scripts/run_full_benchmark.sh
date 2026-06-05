#!/usr/bin/env bash
# Build the dataset and run the SketchMol-style 2-7 property direct benchmark.

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

PYTHON_BIN="${SMMED_PYTHON_BIN:-${PYTHON_BIN:-python}}"
export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
CONDITION_ROWS="$OUTPUT_DIR/condition_rows.csv"
MOLECULE_DB="${SMMED_MOLECULE_DB_CSV:-$OUTPUT_DIR/molecule_database.csv}"
BENCHMARK_OUTPUT_DIR="${SMMED_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_scaffold_retrieval}"
SKIP_BUILD="${SMMED_SKIP_BUILD:-0}"
METHODS="${SMMED_BENCHMARK_METHODS:-source_identity,scaffold_property_retrieval,target_oracle}"
MAX_GLOBAL_CANDIDATES="${SMMED_MAX_GLOBAL_CANDIDATES:-20000}"
SCAFFOLD_FALLBACK_MODE="${SMMED_SCAFFOLD_FALLBACK_MODE:-source_identity}"
SOURCE_TANIMOTO_THRESHOLDS="${SMMED_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"
LIMIT_EVAL_ROWS="${SMMED_LIMIT_EVAL_ROWS:-}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"
COMPUTE_TANIMOTO="${SMMED_COMPUTE_TANIMOTO:-0}"
ALLOW_EVAL_TARGET_CANDIDATES="${SMMED_ALLOW_EVAL_TARGET_CANDIDATES:-0}"
SEED="${SMMED_SEED:-7}"

echo "SketchMol multi-property full benchmark workflow"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  molecule_db=$MOLECULE_DB"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  methods=$METHODS"
echo "  max_eval_per_property_count=$MAX_EVAL_PER_PROPERTY_COUNT"
echo "  scaffold_fallback_mode=$SCAFFOLD_FALLBACK_MODE"
echo "  source_tanimoto_thresholds=$SOURCE_TANIMOTO_THRESHOLDS"

if [[ "$SKIP_BUILD" != "1" ]]; then
  bash "$PROJECT_DIR/scripts/run_build_dataset.sh"
fi

if [[ ! -f "$CONDITION_ROWS" ]]; then
  echo "ERROR: condition rows not found: $CONDITION_ROWS" >&2
  echo "Set SMMED_OUTPUT_DIR or run the dataset build first." >&2
  exit 2
fi
if [[ ! -f "$MOLECULE_DB" ]]; then
  echo "ERROR: molecule database not found: $MOLECULE_DB" >&2
  echo "Set SMMED_OUTPUT_DIR or run the dataset build first." >&2
  exit 2
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT_EVAL_ROWS" ]]; then
  LIMIT_ARGS=(--limit-eval-rows "$LIMIT_EVAL_ROWS")
fi
TANIMOTO_ARGS=()
if [[ "$COMPUTE_TANIMOTO" == "1" ]]; then
  TANIMOTO_ARGS=(--compute-tanimoto)
fi
TARGET_POOL_ARGS=()
if [[ "$ALLOW_EVAL_TARGET_CANDIDATES" == "1" ]]; then
  TARGET_POOL_ARGS=(--allow-eval-target-candidates)
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/benchmark_multiproperty_retrieval.py" \
  --condition-rows-csv "$CONDITION_ROWS" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --candidate-molecule-db-csv "$MOLECULE_DB" \
  --methods "$METHODS" \
  --max-global-candidates "$MAX_GLOBAL_CANDIDATES" \
  --scaffold-fallback-mode "$SCAFFOLD_FALLBACK_MODE" \
  --source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS" \
  --max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT" \
  --seed "$SEED" \
  "${LIMIT_ARGS[@]}" \
  "${TANIMOTO_ARGS[@]}" \
  "${TARGET_POOL_ARGS[@]}"

echo
echo "Multi-property benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo
sed -n '1,80p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
