#!/usr/bin/env bash
# Materialize a real SketchMol + OCR CSV into this benchmark folder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_BENCHMARK_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SKETCHMOL_REPO="${SKETCHMOL_REPO:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
SOURCE_CSV="${SKETCHMOL_BENCHMARK_SOURCE_CSV:-}"
RUN_LOG="${SKETCHMOL_BENCHMARK_RUN_LOG:-}"
OUTPUT_DIR="${SKETCHMOL_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/current}"
BENCHMARK_NAME="${SKETCHMOL_BENCHMARK_NAME:-real_sketchmol_plus_ocr}"

if [[ -z "$SOURCE_CSV" ]]; then
  cat <<EOF >&2
ERROR: SKETCHMOL_BENCHMARK_SOURCE_CSV is required.

It should point to the real SketchMol image_path.csv after running MolScribe:
  Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate/predict_csv.py

Example:
  SKETCHMOL_BENCHMARK_SOURCE_CSV=/path/to/image_path.csv \\
  bash SketchMolBenchmark/scripts/materialize_current.sh
EOF
  exit 2
fi

if [[ ! -f "$SOURCE_CSV" ]]; then
  echo "ERROR: source CSV does not exist: $SOURCE_CSV" >&2
  exit 2
fi

if [[ ! -d "$SKETCHMOL_REPO" ]]; then
  echo "ERROR: SketchMol repo not found: $SKETCHMOL_REPO" >&2
  exit 2
fi

ARGS=(--source-csv "$SOURCE_CSV" --output-dir "$OUTPUT_DIR" --benchmark-name "$BENCHMARK_NAME" --sketchmol-repo "$SKETCHMOL_REPO")
if [[ -n "$RUN_LOG" ]]; then
  ARGS+=(--run-log "$RUN_LOG")
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolBenchmark materialize"
echo "  python=$PYTHON_BIN"
echo "  sketchmol_repo=$SKETCHMOL_REPO"
echo "  source_csv=$SOURCE_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  benchmark_name=$BENCHMARK_NAME"

"$PYTHON_BIN" -m sketchmol_benchmark.materialize "${ARGS[@]}"

echo
echo "SketchMolBenchmark materialized: $OUTPUT_DIR"
echo "  summary=$OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$OUTPUT_DIR/benchmark_decoded.csv"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  report=$OUTPUT_DIR/benchmark_report.md"
