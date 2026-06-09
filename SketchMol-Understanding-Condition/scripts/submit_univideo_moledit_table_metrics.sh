#!/usr/bin/env bash
# Submit MolEdit-Instruct table metrics for SUCC UniVideo benchmark outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v1}"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"

TABLE_REFERENCE="${SUCC_MOLEDIT_TABLE_REFERENCE:-$SUCC_MOLEDIT_EVAL_SPLIT}"
TABLE_PREDICTIONS="${SUCC_MOLEDIT_TABLE_PREDICTIONS:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_primary_fast/benchmark_decoded.csv}"
TABLE_OUTPUT_DIR="${SUCC_MOLEDIT_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics}"
TABLE_METHOD="${SUCC_MOLEDIT_TABLE_METHOD:-edit_latent_source_similarity_rerank}"
TABLE_MODEL_NAME="${SUCC_MOLEDIT_TABLE_MODEL_NAME:-SUCC-UniVideo}"
TABLE_THRESHOLDS="${SUCC_MOLEDIT_TABLE_THRESHOLDS:-0.65,0.15}"
TABLE_TASK_FILTER="${SUCC_MOLEDIT_TABLE_TASK_FILTER:-table1}"
TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-skip-task}"
TABLE_DEPENDENCY="${SUCC_MOLEDIT_TABLE_SLURM_DEPENDENCY:-${SUCC_TABLE_SLURM_DEPENDENCY:-}}"

TABLE_ACCOUNT="${SUCC_MOLEDIT_TABLE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TABLE_TIME="${SUCC_MOLEDIT_TABLE_SLURM_TIME:-00:30:00}"
TABLE_MEM="${SUCC_MOLEDIT_TABLE_SLURM_MEM:-8G}"
TABLE_CPUS="${SUCC_MOLEDIT_TABLE_SLURM_CPUS:-1}"
TABLE_JOB_NAME="${SUCC_MOLEDIT_TABLE_SLURM_JOB_NAME:-succ-moledit-table}"
TABLE_LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
TABLE_PARTITION="${SUCC_MOLEDIT_TABLE_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$TABLE_REFERENCE" ]]; then
  echo "ERROR: missing MolEdit reference split: $TABLE_REFERENCE" >&2
  exit 2
fi
if [[ -z "$TABLE_DEPENDENCY" && ! -f "$TABLE_PREDICTIONS" ]]; then
  echo "ERROR: missing predictions: $TABLE_PREDICTIONS" >&2
  echo "Run the materialized benchmark first, or set SUCC_TABLE_SLURM_DEPENDENCY=afterok:<job_id>." >&2
  exit 2
fi

mkdir -p "$TABLE_LOG_DIR"

TABLE_CMD=(
  "$SUCC_PYTHON_BIN"
  "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/evaluate_moledit_table_metrics.py"
  --reference "$TABLE_REFERENCE"
  --predictions "$TABLE_PREDICTIONS"
  --method "$TABLE_METHOD"
  --output-dir "$TABLE_OUTPUT_DIR"
  --model-name "$TABLE_MODEL_NAME"
  --thresholds "$TABLE_THRESHOLDS"
  --task-filter "$TABLE_TASK_FILTER"
  --missing-oracle-policy "$TABLE_MISSING_ORACLE_POLICY"
)
printf -v TABLE_WRAP_CMD "%q " "${TABLE_CMD[@]}"

TABLE_SBATCH_ARGS=(
  --account="$TABLE_ACCOUNT"
  --job-name="$TABLE_JOB_NAME"
  --time="$TABLE_TIME"
  --mem="$TABLE_MEM"
  --cpus-per-task="$TABLE_CPUS"
  --output="$TABLE_LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$TABLE_PARTITION" ]]; then
  TABLE_SBATCH_ARGS+=(--partition="$TABLE_PARTITION")
fi
if [[ -n "$TABLE_DEPENDENCY" ]]; then
  TABLE_SBATCH_ARGS+=(--dependency="$TABLE_DEPENDENCY")
fi

echo "Submitting SUCC UniVideo MolEdit table metrics"
echo "  reference=$TABLE_REFERENCE"
echo "  predictions=$TABLE_PREDICTIONS"
echo "  method=$TABLE_METHOD"
echo "  output_dir=$TABLE_OUTPUT_DIR"
echo "  missing_oracle_policy=$TABLE_MISSING_ORACLE_POLICY"
if [[ -n "$TABLE_DEPENDENCY" ]]; then
  echo "  dependency=$TABLE_DEPENDENCY"
fi

table_output="$(sbatch "${TABLE_SBATCH_ARGS[@]}" --wrap="$TABLE_WRAP_CMD")"
echo "$table_output"
table_job_id="$(echo "$table_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$table_job_id" ]]; then
  echo "ERROR: failed to parse MolEdit table job id." >&2
  exit 1
fi

echo
echo "MolEdit table metrics submitted."
echo "  job_id=$table_job_id"
echo "  log=$TABLE_LOG_DIR/${TABLE_JOB_NAME}-${table_job_id}.log"
echo "  summary=$TABLE_OUTPUT_DIR/moledit_table_summary.md"
