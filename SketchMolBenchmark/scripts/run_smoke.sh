#!/usr/bin/env bash
# Run lightweight tests for SketchMolBenchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${SKETCHMOL_BENCHMARK_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolBenchmark smoke"
echo "  python=$PYTHON_BIN"

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'

