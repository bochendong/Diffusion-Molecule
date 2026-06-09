#!/usr/bin/env bash
# Merge sharded Unified 3M materialized benchmark outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATASET_PROJECT_DIR="$REPO_DIR/SketchMol-MultiProperty-EditDataset"
cd "$REPO_DIR"

PYTHON_BIN="${SMU3M_PYTHON_BIN:-${SMMED_PYTHON_BIN:-${PYTHON_BIN:-python}}}"
UNIFIED_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1}"
BENCHMARK_PROFILE="${SMU3M_BENCHMARK_PROFILE:-primary_fast}"
case "$BENCHMARK_PROFILE" in
  full) DEFAULT_BENCHMARK_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/benchmark_materialized" ;;
  primary_fast | scaffold) DEFAULT_BENCHMARK_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR/benchmark_materialized_${BENCHMARK_PROFILE}" ;;
  *)
    echo "ERROR: unsupported SMU3M_BENCHMARK_PROFILE=$BENCHMARK_PROFILE" >&2
    exit 2
    ;;
esac
BENCHMARK_OUTPUT_DIR="${SMU3M_BENCHMARK_OUTPUT_DIR:-$DEFAULT_BENCHMARK_OUTPUT_DIR}"
SHARDS_DIR="${SMU3M_BENCHMARK_SHARDS_DIR:-$BENCHMARK_OUTPUT_DIR/shards}"
SOURCE_TANIMOTO_THRESHOLDS="${SMMED_SOURCE_TANIMOTO_THRESHOLDS:-}"

export PYTHONPATH="$DATASET_PROJECT_DIR/scripts:$PROJECT_DIR:$DATASET_PROJECT_DIR:$REPO_DIR/SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"

MERGE_ARGS=(
  --shards-dir "$SHARDS_DIR"
  --output-dir "$BENCHMARK_OUTPUT_DIR"
)
if [[ -n "$SOURCE_TANIMOTO_THRESHOLDS" ]]; then
  MERGE_ARGS+=(--source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS")
fi

"$PYTHON_BIN" "$DATASET_PROJECT_DIR/scripts/merge_benchmark_shards.py" "${MERGE_ARGS[@]}"

echo
echo "Merged materialized benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
