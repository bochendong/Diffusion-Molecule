#!/usr/bin/env bash
# Submit a fairer Table1 rerun for the v3 MolEdit checkpoint using the primary-fast
# source-similarity materializer instead of table-success reranking.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"

# The historical v3 attack checkpoint was written into the v2_fix output tree.
# Override SUCC_UNIFIED_OUTPUT_DIR if a clean v3 directory exists on the cluster.
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix}"
export SUCC_MOLEDIT_TRAIN_SPLIT="${SUCC_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS="${SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS:-1}"
export SUCC_TABLE1_PACK_DIR="${SUCC_TABLE1_PACK_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/dataset/table1_benchmark_fair}"
export SUCC_TABLE1_EVAL_OUTPUT_DIR="${SUCC_TABLE1_EVAL_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/table1_eval_latent_fair}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
export SUCC_TABLE1_BENCHMARK_OUTPUT_DIR="${SUCC_TABLE1_BENCHMARK_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_table1_fair}"
export SUCC_TABLE1_TABLE_OUTPUT_DIR="${SUCC_TABLE1_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics_table1_fair}"
export SUCC_MOLEDIT_TABLE_METHOD="${SUCC_MOLEDIT_TABLE_METHOD:-edit_latent_source_similarity_rerank}"
export SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1="${SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1:-1}"
export SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE="${SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE:-1}"
export SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-fail}"
export SUCC_TABLE1_EVAL_SLURM_JOB_NAME="${SUCC_TABLE1_EVAL_SLURM_JOB_NAME:-succ-table1-eval-fair}"
export SUCC_TABLE1_BENCH_SLURM_JOB_NAME="${SUCC_TABLE1_BENCH_SLURM_JOB_NAME:-succ-table1-bench-fair}"
export SUCC_TABLE1_TABLE_SLURM_JOB_NAME="${SUCC_TABLE1_TABLE_SLURM_JOB_NAME:-succ-table1-table-fair}"

bash "$SCRIPT_DIR/submit_univideo_moledit_table1_extension.sh"
