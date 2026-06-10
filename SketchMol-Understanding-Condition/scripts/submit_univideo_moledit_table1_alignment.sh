#!/usr/bin/env bash
# Submit strict MolEditRL Table1-aligned metrics for an existing SUCC UniVideo run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix}"
export SUCC_MOLEDIT_TABLE_REFERENCE="${SUCC_MOLEDIT_TABLE_REFERENCE:-$SUCC_UNIFIED_OUTPUT_DIR/dataset/univideo_edit_eval.jsonl}"
export SUCC_MOLEDIT_TABLE_PREDICTIONS="${SUCC_MOLEDIT_TABLE_PREDICTIONS:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_primary_fast/benchmark_decoded.csv}"
export SUCC_MOLEDIT_TABLE_OUTPUT_DIR="${SUCC_MOLEDIT_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table1_alignment_metrics}"
export SUCC_MOLEDIT_TABLE_METHOD="${SUCC_MOLEDIT_TABLE_METHOD:-edit_latent_source_similarity_rerank}"
export SUCC_MOLEDIT_TABLE_MODEL_NAME="${SUCC_MOLEDIT_TABLE_MODEL_NAME:-SUCC-v2-fix}"
export SUCC_MOLEDIT_TABLE_TASK_FILTER="${SUCC_MOLEDIT_TABLE_TASK_FILTER:-table1}"
export SUCC_MOLEDIT_TABLE_THRESHOLDS="${SUCC_MOLEDIT_TABLE_THRESHOLDS:-0.65,0.15}"
export SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1=1
export SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE="${SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE:-1}"
export SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-fail}"
export SUCC_MOLEDIT_TABLE_SLURM_JOB_NAME="${SUCC_MOLEDIT_TABLE_SLURM_JOB_NAME:-succ-table1-align}"

echo "Submitting strict SUCC Table1 alignment metrics"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  reference=$SUCC_MOLEDIT_TABLE_REFERENCE"
echo "  predictions=$SUCC_MOLEDIT_TABLE_PREDICTIONS"
echo "  table_output=$SUCC_MOLEDIT_TABLE_OUTPUT_DIR"
echo "  missing_oracle_policy=$SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY"
echo "  require_table1_coverage=$SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE"

bash "$SCRIPT_DIR/submit_univideo_moledit_table_metrics.sh"
