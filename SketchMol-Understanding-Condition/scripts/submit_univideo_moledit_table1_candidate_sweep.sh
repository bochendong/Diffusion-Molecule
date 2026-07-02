#!/usr/bin/env bash
# Submit a MolEdit Table1 candidate-budget sweep for table-success reranking.
#
# Default sweep:
#   n=20   - aligned with MuMO's 20 candidates per input
#   n=256  - medium candidate-pool ablation
#
# The script exports/reuses the Table1 benchmark pack, reuses existing eval
# latents when available, and submits one materialized benchmark plus one
# MolEdit table-metrics job for each candidate budget.

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

# Match the v3 attack/Table1-complete setup used for the 2048-candidate row.
export SUCC_TABLE1_PER_TASK="${SUCC_TABLE1_PER_TASK:-100}"
export SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS="${SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS:-1}"
export SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO="${SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO:-0.4}"
export SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT="${SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT:-8000}"
export SUCC_TABLE1_PACK_DIR="${SUCC_TABLE1_PACK_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/dataset/table1_benchmark_synthetic}"
export SUCC_TABLE1_EVAL_OUTPUT_DIR="${SUCC_TABLE1_EVAL_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/table1_eval_latent}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-table_attack}"
export SUCC_MOLEDIT_TABLE_METHOD="${SUCC_MOLEDIT_TABLE_METHOD:-edit_latent_table_success_rerank}"
export SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-fail}"
export SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1="${SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1:-1}"
export SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE="${SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE:-1}"

# Keep latent inference defaults aligned with the v3 attack run if eval must be
# regenerated. Set SUCC_TABLE1_REUSE_EVAL=1 to force reuse, or 0 to force rerun.
export SUCC_CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v3_attack}"
export SUCC_SAMPLE_STEPS="${SUCC_SAMPLE_STEPS:-48}"
export SUCC_EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"

export SUCC_TABLE_SUCCESS_WEIGHT="${SUCC_TABLE_SUCCESS_WEIGHT:-100.0}"
export SUCC_TABLE_SOURCE_WEIGHT="${SUCC_TABLE_SOURCE_WEIGHT:-6.0}"
export SUCC_TABLE_LATENT_WEIGHT="${SUCC_TABLE_LATENT_WEIGHT:-1.0}"

SWEEP_RAW="${SUCC_TABLE1_CANDIDATE_SWEEP:-20 256}"
SWEEP_RAW="${SWEEP_RAW//,/ }"
BUDGETS=()
for budget in $SWEEP_RAW; do
  if ! [[ "$budget" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid candidate budget '$budget' in SUCC_TABLE1_CANDIDATE_SWEEP." >&2
    exit 2
  fi
  if [[ "$budget" -lt 1 ]]; then
    echo "ERROR: candidate budget must be positive: $budget" >&2
    exit 2
  fi
  BUDGETS+=("$budget")
done
if [[ "${#BUDGETS[@]}" -eq 0 ]]; then
  echo "ERROR: no candidate budgets provided." >&2
  exit 2
fi

SLURM_ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
SLURM_PARTITION="${SUCC_SLURM_PARTITION:-}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
TABLE1_REFERENCE="${SUCC_TABLE1_REFERENCE:-$SUCC_TABLE1_PACK_DIR/table1_moledit_rows.csv}"
TABLE1_EVAL_JSONL="${SUCC_TABLE1_EVAL_JSONL:-$SUCC_TABLE1_PACK_DIR/table1_eval.jsonl}"
TABLE1_BENCHMARK_ROWS="${SUCC_TABLE1_BENCHMARK_ROWS_CSV:-$SUCC_TABLE1_PACK_DIR/table1_benchmark_condition_rows.csv}"
EVAL_LATENT_DIR="$SUCC_TABLE1_EVAL_OUTPUT_DIR/eval_latent"
PREDICTIONS_CSV="$EVAL_LATENT_DIR/predictions.csv"
GENERATED_LATENTS="$EVAL_LATENT_DIR/generated_latents.npy"
CANDIDATE_LATENTS="$EVAL_LATENT_DIR/target_latents.npy"
OUTPUT_ROOT="${SUCC_TABLE1_SWEEP_OUTPUT_ROOT:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule}"
EXPORT_PACK_MODE="${SUCC_TABLE1_EXPORT_PACK:-auto}"
REUSE_EVAL_MODE="${SUCC_TABLE1_REUSE_EVAL:-auto}"
MATCH_SOURCE_BUDGET="${SUCC_TABLE1_MATCH_SOURCE_RERANK_BUDGET:-1}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$SUCC_TABLE1_PACK_DIR" "$LOG_DIR"

pack_ready=1
for required in "$TABLE1_REFERENCE" "$TABLE1_EVAL_JSONL" "$TABLE1_BENCHMARK_ROWS"; do
  if [[ ! -f "$required" ]]; then
    pack_ready=0
  fi
done

run_pack_export=0
case "$EXPORT_PACK_MODE" in
  1 | true | yes) run_pack_export=1 ;;
  0 | false | no) run_pack_export=0 ;;
  auto)
    if [[ "$pack_ready" == "0" ]]; then
      run_pack_export=1
    fi
    ;;
  *)
    echo "ERROR: unsupported SUCC_TABLE1_EXPORT_PACK=$EXPORT_PACK_MODE (use auto, 1, or 0)." >&2
    exit 2
    ;;
esac

echo "MolEdit Table1 candidate-budget sweep"
echo "  unified_output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  budgets=${BUDGETS[*]}"
echo "  table1_pack_dir=$SUCC_TABLE1_PACK_DIR"
echo "  export_pack=$EXPORT_PACK_MODE"
echo "  reuse_eval=$REUSE_EVAL_MODE"
echo "  profile=$SUCC_MATERIALIZED_BENCHMARK_PROFILE"
echo "  table_method=$SUCC_MOLEDIT_TABLE_METHOD"
echo "  table_success_weight=$SUCC_TABLE_SUCCESS_WEIGHT"
echo "  table_source_weight=$SUCC_TABLE_SOURCE_WEIGHT"
echo "  output_root=$OUTPUT_ROOT"

if [[ "$run_pack_export" == "1" ]]; then
  echo
  echo "Exporting Table1 benchmark pack"
  EXPORT_ARGS=(
    "$SUCC_PYTHON_BIN"
    "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/export_moledit_table1_benchmark_pack.py"
    --moledit-train-split "$SUCC_MOLEDIT_TRAIN_SPLIT"
    --moledit-eval-split "$SUCC_MOLEDIT_EVAL_SPLIT"
    --output-dir "$SUCC_TABLE1_PACK_DIR"
    --per-task "$SUCC_TABLE1_PER_TASK"
  )
  if [[ "$SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS" == "1" ]]; then
    EXPORT_ARGS+=(
      --synthesize-missing-tasks
      --synthetic-min-source-tanimoto "$SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO"
      --synthetic-candidate-limit "$SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT"
    )
  fi
  "${EXPORT_ARGS[@]}"
else
  echo
  echo "Reusing existing Table1 benchmark pack"
fi

for required in "$TABLE1_REFERENCE" "$TABLE1_EVAL_JSONL" "$TABLE1_BENCHMARK_ROWS"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required Table1 pack artifact: $required" >&2
    exit 2
  fi
done

eval_ready=1
for required in "$PREDICTIONS_CSV" "$GENERATED_LATENTS" "$CANDIDATE_LATENTS"; do
  if [[ ! -f "$required" ]]; then
    eval_ready=0
  fi
done

reuse_eval=0
case "$REUSE_EVAL_MODE" in
  1 | true | yes) reuse_eval=1 ;;
  0 | false | no) reuse_eval=0 ;;
  auto)
    if [[ "$eval_ready" == "1" ]]; then
      reuse_eval=1
    fi
    ;;
  *)
    echo "ERROR: unsupported SUCC_TABLE1_REUSE_EVAL=$REUSE_EVAL_MODE (use auto, 1, or 0)." >&2
    exit 2
    ;;
esac
if [[ "$reuse_eval" == "1" && "$eval_ready" == "0" ]]; then
  echo "ERROR: SUCC_TABLE1_REUSE_EVAL=$REUSE_EVAL_MODE but eval latent artifacts are missing." >&2
  echo "       Missing artifacts should be under: $EVAL_LATENT_DIR" >&2
  exit 2
fi

eval_job_id="reuse"
bench_dependency=""
if [[ "$reuse_eval" == "1" ]]; then
  echo
  echo "Reusing existing Table1 eval latents"
  echo "  predictions=$PREDICTIONS_CSV"
  echo "  generated_latents=$GENERATED_LATENTS"
  echo "  candidate_latents=$CANDIDATE_LATENTS"
else
  EVAL_TIME="${SUCC_TABLE1_EVAL_SLURM_TIME:-03:00:00}"
  EVAL_MEM="${SUCC_TABLE1_EVAL_SLURM_MEM:-16G}"
  EVAL_CPUS="${SUCC_TABLE1_EVAL_SLURM_CPUS:-1}"
  EVAL_JOB_NAME="${SUCC_TABLE1_EVAL_SLURM_JOB_NAME:-succ-table1-eval}"
  EVAL_DEPENDENCY="${SUCC_TABLE1_EVAL_SLURM_DEPENDENCY:-}"
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
  if [[ -n "$EVAL_DEPENDENCY" ]]; then
    eval_sbatch_args+=(--dependency="$EVAL_DEPENDENCY")
  fi

  echo
  echo "Submitting Table1 eval_latent job"
  echo "  output_dir=$SUCC_TABLE1_EVAL_OUTPUT_DIR"
  echo "  sample_steps=$SUCC_SAMPLE_STEPS"
  if [[ -n "$EVAL_DEPENDENCY" ]]; then
    echo "  dependency=$EVAL_DEPENDENCY"
  fi
  eval_output="$(sbatch "${eval_sbatch_args[@]}" --wrap="bash '$SCRIPT_DIR/run_univideo_table1_eval_latent.sh'")"
  echo "$eval_output"
  eval_job_id="$(echo "$eval_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$eval_job_id" ]]; then
    echo "ERROR: failed to parse Table1 eval job id." >&2
    exit 1
  fi
  bench_dependency="afterok:$eval_job_id"
fi

BENCH_TIME="${SUCC_TABLE1_BENCH_SLURM_TIME:-04:00:00}"
BENCH_MEM="${SUCC_TABLE1_BENCH_SLURM_MEM:-16G}"
BENCH_CPUS="${SUCC_TABLE1_BENCH_SLURM_CPUS:-1}"
TABLE_SUMMARIES=()

for budget in "${BUDGETS[@]}"; do
  benchmark_output_dir="$OUTPUT_ROOT/benchmark_materialized_table1_${SUCC_MATERIALIZED_BENCHMARK_PROFILE}_n${budget}"
  table_output_dir="$OUTPUT_ROOT/moledit_table_metrics_table1_${SUCC_MATERIALIZED_BENCHMARK_PROFILE}_n${budget}"
  bench_job_name="${SUCC_TABLE1_BENCH_SLURM_JOB_NAME_PREFIX:-succ-t1-bench}-n${budget}"
  table_job_name="${SUCC_TABLE1_TABLE_SLURM_JOB_NAME_PREFIX:-succ-t1-table}-n${budget}"

  source_rerank_candidates="${SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES:-}"
  if [[ "$MATCH_SOURCE_BUDGET" == "1" || -z "$source_rerank_candidates" ]]; then
    source_rerank_candidates="$budget"
  fi

  echo
  echo "Submitting Table1 materialized benchmark for n=$budget"
  echo "  table_success_rerank_candidates=$budget"
  echo "  source_similarity_rerank_candidates=$source_rerank_candidates"
  echo "  benchmark_output_dir=$benchmark_output_dir"
  if [[ -n "$bench_dependency" ]]; then
    echo "  dependency=$bench_dependency"
  fi

  bench_output="$(
    SUCC_EVAL_LATENT_DIR="$EVAL_LATENT_DIR" \
    SUCC_PREDICTIONS_CSV="$PREDICTIONS_CSV" \
    SUCC_GENERATED_LATENTS="$GENERATED_LATENTS" \
    SUCC_CANDIDATE_LATENTS="$CANDIDATE_LATENTS" \
    SUCC_EVAL_JSONL="$TABLE1_EVAL_JSONL" \
    SUCC_BENCHMARK_ROWS_CSV="$TABLE1_BENCHMARK_ROWS" \
    SUCC_MATERIALIZED_BENCHMARK_OUTPUT_DIR="$benchmark_output_dir" \
    SUCC_MATERIALIZED_BENCHMARK_PROFILE="$SUCC_MATERIALIZED_BENCHMARK_PROFILE" \
    SUCC_MATERIALIZED_SLURM_DEPENDENCY="$bench_dependency" \
    SUCC_MATERIALIZED_SLURM_JOB_NAME="$bench_job_name" \
    SUCC_MATERIALIZED_SLURM_TIME="$BENCH_TIME" \
    SUCC_MATERIALIZED_SLURM_MEM="$BENCH_MEM" \
    SUCC_MATERIALIZED_SLURM_CPUS="$BENCH_CPUS" \
    SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES="$source_rerank_candidates" \
    SUCC_TABLE_SUCCESS_RERANK_CANDIDATES="$budget" \
    SUCC_TABLE_SUCCESS_WEIGHT="$SUCC_TABLE_SUCCESS_WEIGHT" \
    SUCC_TABLE_SOURCE_WEIGHT="$SUCC_TABLE_SOURCE_WEIGHT" \
    SUCC_TABLE_LATENT_WEIGHT="$SUCC_TABLE_LATENT_WEIGHT" \
    bash "$SCRIPT_DIR/submit_univideo_materialized_benchmark.sh"
  )"
  echo "$bench_output"
  bench_job_id="$(echo "$bench_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$bench_job_id" ]]; then
    echo "ERROR: failed to parse Table1 benchmark job id for n=$budget." >&2
    exit 1
  fi

  table_dependency="afterok:$bench_job_id"
  echo
  echo "Submitting Table1 MolEdit table metrics for n=$budget"
  echo "  dependency=$table_dependency"
  echo "  table_output_dir=$table_output_dir"

  table_output="$(
    SUCC_MOLEDIT_TABLE_REFERENCE="$TABLE1_REFERENCE" \
    SUCC_MOLEDIT_TABLE_PREDICTIONS="$benchmark_output_dir/benchmark_decoded.csv" \
    SUCC_MOLEDIT_TABLE_OUTPUT_DIR="$table_output_dir" \
    SUCC_MOLEDIT_TABLE_METHOD="$SUCC_MOLEDIT_TABLE_METHOD" \
    SUCC_MOLEDIT_TABLE_SLURM_DEPENDENCY="$table_dependency" \
    SUCC_MOLEDIT_TABLE_SLURM_JOB_NAME="$table_job_name" \
    SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="$SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY" \
    SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1="$SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1" \
    SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE="$SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE" \
    bash "$SCRIPT_DIR/submit_univideo_moledit_table_metrics.sh"
  )"
  echo "$table_output"
  table_job_id="$(echo "$table_output" | sed -n 's/.*job_id=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$table_job_id" ]]; then
    table_job_id="$(echo "$table_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  fi
  if [[ -z "$table_job_id" ]]; then
    echo "ERROR: failed to parse Table1 table metrics job id for n=$budget." >&2
    exit 1
  fi
  TABLE_SUMMARIES+=("n=$budget benchmark_job=$bench_job_id table_job=$table_job_id summary=$table_output_dir/moledit_table_summary.md")
done

echo
echo "MolEdit Table1 candidate-budget sweep submitted."
echo "  eval_job_id=$eval_job_id"
for line in "${TABLE_SUMMARIES[@]}"; do
  echo "  $line"
done
