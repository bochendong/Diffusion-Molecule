#!/usr/bin/env bash
# Materialize direct image-to-structure predictions with SketchMol benchmark metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_BENCHMARK_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SOURCE_CSV="${SKETCHMOL_DIRECT_PREDICTIONS_CSV:-${SKETCHMOL_BENCHMARK_SOURCE_CSV:-}}"
OUTPUT_DIR="${SKETCHMOL_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/direct_structure_current}"
BENCHMARK_NAME="${SKETCHMOL_BENCHMARK_NAME:-direct_structure_prediction}"
SMILES_COLUMN="${SKETCHMOL_DIRECT_SMILES_COLUMN:-generated_smiles}"
IMAGE_COLUMN="${SKETCHMOL_DIRECT_IMAGE_COLUMN:-image_path}"
RUN_LOG="${SKETCHMOL_BENCHMARK_RUN_LOG:-}"

if [[ -z "$SOURCE_CSV" ]]; then
  cat <<EOF >&2
ERROR: SKETCHMOL_DIRECT_PREDICTIONS_CSV is required.

It should point to a CSV with SketchMol condition columns plus a prediction
column such as generated_smiles.

Example:
  SKETCHMOL_DIRECT_PREDICTIONS_CSV=/path/to/direct_predictions.csv \\
  SKETCHMOL_DIRECT_SMILES_COLUMN=generated_smiles \\
  bash SketchMolBenchmark/scripts/materialize_direct_predictions.sh
EOF
  exit 2
fi

if [[ ! -f "$SOURCE_CSV" ]]; then
  echo "ERROR: direct prediction CSV does not exist: $SOURCE_CSV" >&2
  exit 2
fi

ARGS=(
  --source-csv "$SOURCE_CSV"
  --output-dir "$OUTPUT_DIR"
  --benchmark-name "$BENCHMARK_NAME"
  --benchmark-kind "direct_structure_prediction"
  --benchmark-task "direct_structure_prediction"
  --smiles-column "$SMILES_COLUMN"
  --image-column "$IMAGE_COLUMN"
  --present-rate-field "predicted_smiles_present_rate"
  --present-detail-field "predicted_smiles_present"
  --source-copy-name "source_direct_predictions.csv"
  --report-title "Direct Structure Prediction Benchmark"
  --present-label "predicted SMILES present"
  --report-description "This artifact evaluates a direct structure decoder on SketchMol-compatible condition rows, using the same validity, property-success, and MAE logic as the real SketchMol+OCR materialization."
)
if [[ -n "$RUN_LOG" ]]; then
  ARGS+=(--run-log "$RUN_LOG")
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolBenchmark direct prediction materialize"
echo "  python=$PYTHON_BIN"
echo "  source_csv=$SOURCE_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  benchmark_name=$BENCHMARK_NAME"
echo "  smiles_column=$SMILES_COLUMN"
echo "  image_column=$IMAGE_COLUMN"

"$PYTHON_BIN" -m sketchmol_benchmark.materialize "${ARGS[@]}"

echo
echo "Direct prediction benchmark materialized: $OUTPUT_DIR"
echo "  summary=$OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$OUTPUT_DIR/benchmark_decoded.csv"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  report=$OUTPUT_DIR/benchmark_report.md"
