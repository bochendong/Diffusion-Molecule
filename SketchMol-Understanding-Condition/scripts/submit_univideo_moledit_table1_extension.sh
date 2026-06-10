#!/usr/bin/env bash
# Export Table1-balanced benchmark pack, run eval_latent, materialized benchmark, and table metrics.

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
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix}"
export SUCC_MOLEDIT_TRAIN_SPLIT="${SUCC_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SUCC_TABLE1_PER_TASK="${SUCC_TABLE1_PER_TASK:-100}"
export SUCC_TABLE1_PACK_DIR="${SUCC_TABLE1_PACK_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/dataset/table1_benchmark}"
export SUCC_TABLE1_EVAL_OUTPUT_DIR="${SUCC_TABLE1_EVAL_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/table1_eval_latent}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
export SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-fail}"

TABLE1_BENCHMARK_OUTPUT_DIR="${SUCC_TABLE1_BENCHMARK_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_table1_primary_fast}"
TABLE1_TABLE_OUTPUT_DIR="${SUCC_TABLE1_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics_table1}"
TABLE1_REFERENCE="${SUCC_TABLE1_REFERENCE:-$SUCC_TABLE1_PACK_DIR/table1_moledit_rows.csv}"

SLURM_ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
SLURM_PARTITION="${SUCC_SLURM_PARTITION:-}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$SUCC_TABLE1_PACK_DIR" "$LOG_DIR"

echo "Exporting Table1 benchmark pack"
echo "  train_split=$SUCC_MOLEDIT_TRAIN_SPLIT"
echo "  eval_split=$SUCC_MOLEDIT_EVAL_SPLIT"
echo "  per_task=$SUCC_TABLE1_PER_TASK"
echo "  output_dir=$SUCC_TABLE1_PACK_DIR"

"$SUCC_PYTHON_BIN" "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/export_moledit_table1_benchmark_pack.py" \
  --moledit-train-split "$SUCC_MOLEDIT_TRAIN_SPLIT" \
  --moledit-eval-split "$SUCC_MOLEDIT_EVAL_SPLIT" \
  --output-dir "$SUCC_TABLE1_PACK_DIR" \
  --per-task "$SUCC_TABLE1_PER_TASK"

EVAL_TIME="${SUCC_TABLE1_EVAL_SLURM_TIME:-03:00:00}"
EVAL_MEM="${SUCC_TABLE1_EVAL_SLURM_MEM:-16G}"
EVAL_CPUS="${SUCC_TABLE1_EVAL_SLURM_CPUS:-1}"
EVAL_JOB_NAME="${SUCC_TABLE1_EVAL_SLURM_JOB_NAME:-succ-table1-eval}"

eval_sbatch_args=(
  --account="$SLURM_ACCOUNT"
  --job-name="$EVAL_JOB_NAME"
  --time="$EVAL_TIME"
  --mem="$EVAL_MEM"
  --cpus-per-task="$EVAL_CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$SLURM_PARTITION" ]]; then
  eval_sbatch_args+=(--partition="$SLURM_PARTITION")
fi

echo
echo "Submitting Table1 eval_latent job"
eval_output="$(sbatch "${eval_sbatch_args[@]}" --wrap="bash '$SCRIPT_DIR/run_univideo_table1_eval_latent.sh'")"
echo "$eval_output"
eval_job_id="$(echo "$eval_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$eval_job_id" ]]; then
  echo "ERROR: failed to parse Table1 eval job id." >&2
  exit 1
fi

BENCH_TIME="${SUCC_TABLE1_BENCH_SLURM_TIME:-04:00:00}"
BENCH_MEM="${SUCC_TABLE1_BENCH_SLURM_MEM:-16G}"
BENCH_CPUS="${SUCC_TABLE1_BENCH_SLURM_CPUS:-1}"
BENCH_JOB_NAME="${SUCC_TABLE1_BENCH_SLURM_JOB_NAME:-succ-table1-bench}"
bench_dependency="afterok:$eval_job_id"

echo
echo "Submitting Table1 materialized benchmark"
echo "  dependency=$bench_dependency"
echo "  benchmark_output_dir=$TABLE1_BENCHMARK_OUTPUT_DIR"

bench_output="$(
  SUCC_EVAL_LATENT_DIR="$SUCC_TABLE1_EVAL_OUTPUT_DIR/eval_latent" \
  SUCC_PREDICTIONS_CSV="$SUCC_TABLE1_EVAL_OUTPUT_DIR/eval_latent/predictions.csv" \
  SUCC_GENERATED_LATENTS="$SUCC_TABLE1_EVAL_OUTPUT_DIR/eval_latent/generated_latents.npy" \
  SUCC_CANDIDATE_LATENTS="$SUCC_TABLE1_EVAL_OUTPUT_DIR/eval_latent/target_latents.npy" \
  SUCC_EVAL_JSONL="$SUCC_TABLE1_PACK_DIR/table1_eval.jsonl" \
  SUCC_BENCHMARK_ROWS_CSV="$SUCC_TABLE1_PACK_DIR/table1_benchmark_condition_rows.csv" \
  SUCC_MATERIALIZED_BENCHMARK_OUTPUT_DIR="$TABLE1_BENCHMARK_OUTPUT_DIR" \
  SUCC_MATERIALIZED_SLURM_DEPENDENCY="$bench_dependency" \
  SUCC_MATERIALIZED_SLURM_JOB_NAME="$BENCH_JOB_NAME" \
  SUCC_MATERIALIZED_SLURM_TIME="$BENCH_TIME" \
  SUCC_MATERIALIZED_SLURM_MEM="$BENCH_MEM" \
  SUCC_MATERIALIZED_SLURM_CPUS="$BENCH_CPUS" \
  bash "$SCRIPT_DIR/submit_univideo_materialized_benchmark.sh"
)"
echo "$bench_output"
bench_job_id="$(echo "$bench_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$bench_job_id" ]]; then
  echo "ERROR: failed to parse Table1 benchmark job id." >&2
  exit 1
fi

table_dependency="afterok:$bench_job_id"
echo
echo "Submitting Table1 MolEdit table metrics"
echo "  dependency=$table_dependency"
echo "  reference=$TABLE1_REFERENCE"

SUCC_MOLEDIT_TABLE_REFERENCE="$TABLE1_REFERENCE" \
SUCC_MOLEDIT_TABLE_PREDICTIONS="$TABLE1_BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv" \
SUCC_MOLEDIT_TABLE_OUTPUT_DIR="$TABLE1_TABLE_OUTPUT_DIR" \
SUCC_MOLEDIT_TABLE_SLURM_DEPENDENCY="$table_dependency" \
SUCC_MOLEDIT_TABLE_SLURM_JOB_NAME="${SUCC_TABLE1_TABLE_SLURM_JOB_NAME:-succ-table1-table}" \
SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="$SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY" \
bash "$SCRIPT_DIR/submit_univideo_moledit_table_metrics.sh"

echo
echo "Table1 extension workflow submitted."
echo "  pack_dir=$SUCC_TABLE1_PACK_DIR"
echo "  eval_job_id=$eval_job_id"
echo "  benchmark_job_id=$bench_job_id"
echo "  table_summary=$TABLE1_TABLE_OUTPUT_DIR/moledit_table_summary.md"
