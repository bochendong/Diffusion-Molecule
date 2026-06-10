#!/usr/bin/env bash
# Submit materialized MolEdit benchmark for the source-anchored Unified 3M v2 run.

set -euo pipefail

export SMU3M_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_sourceanchored_v2}"
export SMU3M_MOLEDIT_TABLE_OUTPUT_DIR="${SMU3M_MOLEDIT_TABLE_OUTPUT_DIR:-$SMU3M_OUTPUT_DIR/moledit_table_metrics_sourceanchored_v2}"

bash "$(dirname "${BASH_SOURCE[0]}")/submit_unified_moledit_benchmark.sh"
