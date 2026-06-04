#!/usr/bin/env bash
# Summarize trajectory and run metrics for paper tables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LETA_PYTHON_BIN:-${PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}}"
TRAJECTORY_PATH="${LETA_TRAJECTORY_PATH:-outputs/trajectories/sketchmol_opt_bootstrap.jsonl}"
RUN_GLOB="${LETA_RUN_GLOB:-outputs/runs/sketchmol_trajectory_suite_seed7_*}"
OUTPUT_JSON="${LETA_METRICS_JSON:-outputs/metrics/paper_metrics_summary.json}"
OUTPUT_CSV="${LETA_METRICS_CSV:-outputs/metrics/paper_run_summary.csv}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m latent_edit_trajectory_attention.metrics \
  --trajectory-path "$TRAJECTORY_PATH" \
  --run-glob "$RUN_GLOB" \
  --output-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV"

