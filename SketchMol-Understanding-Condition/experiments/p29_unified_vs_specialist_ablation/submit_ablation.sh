#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24="$PROJECT/experiments/p24_molprogram_instruct_4m"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
OUTPUT_ROOT="${P29_OUTPUT_ROOT:-$PROJECT/outputs/p29_unified_vs_specialist_ablation/seed_24003}"
LOG_DIR="$PROJECT/logs/p29_unified_vs_specialist_ablation"
ACCOUNT="${P29_ACCOUNT:-${P24_ACCOUNT:-def-hup-ab}}"
TRAIN_GRES="${P29_TRAIN_GRES:-gpu:h100:1}"
EVAL_GRES="${P29_EVAL_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"

unified_adapter="$P24_OUT/full/adapter"
unified_required="$P24_OUT/full/TRAINING_COMPLETE"
test -f "$unified_adapter/adapter_model.safetensors"
test -f "$unified_required"

construction_job=$(sbatch --parsable --account="$ACCOUNT" --job-name=p29-construct \
  --time=1-18:00:00 --cpus-per-task=6 --mem=64G --gres="$TRAIN_GRES" \
  --output="$LOG_DIR/construction-train-%j.log" \
  --export=ALL,P29_SCRIPT_DIR="$SCRIPT_DIR",P29_OUTPUT_ROOT="$OUTPUT_ROOT",P29_SPECIALIST_MODE=construction \
  "$SCRIPT_DIR/run_specialist_train.sh")
editing_job=$(sbatch --parsable --account="$ACCOUNT" --job-name=p29-edit \
  --time=1-18:00:00 --cpus-per-task=6 --mem=64G --gres="$TRAIN_GRES" \
  --output="$LOG_DIR/editing-train-%j.log" \
  --export=ALL,P29_SCRIPT_DIR="$SCRIPT_DIR",P29_OUTPUT_ROOT="$OUTPUT_ROOT",P29_SPECIALIST_MODE=editing \
  "$SCRIPT_DIR/run_specialist_train.sh")

submit_table1() {
  local arm="$1" adapter="$2" required="$3" dependency="$4" train_data="$5"
  local out="$OUTPUT_ROOT/$arm/eval_table1" dep_args=()
  [[ -n "$dependency" ]] && dep_args+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  local generation scoring
  generation=$(sbatch --parsable --account="$ACCOUNT" --job-name="p29-${arm}-t1g" \
    --time=06:00:00 --cpus-per-task=4 --mem=48G --gres="$EVAL_GRES" "${dep_args[@]}" \
    --output="$LOG_DIR/${arm}-table1-generate-%j.log" \
    --export=ALL,P24_SCRIPT_DIR="$P24",P24_EVAL_ADAPTER="$adapter",P24_EVAL_REQUIRED="$required",P24_EVAL_LABEL="p29_${arm}",P24_TABLE1_OUT="$out" \
    "$P24/run_table1_generate.sh")
  scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name="p29-${arm}-t1s" \
    --time=01:00:00 --cpus-per-task=4 --mem=24G \
    --dependency="afterok:$generation" --kill-on-invalid-dep=yes \
    --output="$LOG_DIR/${arm}-table1-score-%j.log" \
    --export=ALL,P24_SCRIPT_DIR="$P24",P24_TABLE1_OUT="$out",P24_TABLE1_MODEL_NAME="P29-$arm",P24_TABLE1_TRAIN_DATA="$train_data",P24_TABLE1_PROTOCOL="p29_${arm}_denovo_best40_v1" \
    "$P24/run_table1_finalize.sh")
  printf '%s %s\n' "$generation" "$scoring"
}

submit_table2() {
  local arm="$1" adapter="$2" required="$3" dependency="$4"
  local out="$OUTPUT_ROOT/$arm/eval_table2" dep_args=()
  [[ -n "$dependency" ]] && dep_args+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  local generation scoring
  generation=$(sbatch --parsable --account="$ACCOUNT" --job-name="p29-${arm}-t2g" \
    --time=03:00:00 --cpus-per-task=4 --mem=48G --gres="$EVAL_GRES" "${dep_args[@]}" \
    --output="$LOG_DIR/${arm}-table2-generate-%j.log" \
    --export=ALL,P24_SCRIPT_DIR="$P24",P24_EVAL_ADAPTER="$adapter",P24_EVAL_REQUIRED="$required",P24_EVAL_LABEL="p29_${arm}_raw1",P24_TABLE2_OUT="$out" \
    "$P24/run_table2_generate.sh")
  scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name="p29-${arm}-t2s" \
    --time=01:00:00 --cpus-per-task=4 --mem=24G \
    --dependency="afterok:$generation" --kill-on-invalid-dep=yes \
    --output="$LOG_DIR/${arm}-table2-score-%j.log" \
    --export=ALL,P24_SCRIPT_DIR="$P24",P24_TABLE2_OUT="$out",P24_TABLE2_MODEL_NAME="P29-$arm-Raw1" \
    "$P24/run_table2_score.sh")
  printf '%s %s\n' "$generation" "$scoring"
}

read -r unified_t1g unified_t1s <<<"$(submit_table1 unified "$unified_adapter" "$unified_required" "" "488,490/569,905")"
read -r unified_t2g unified_t2s <<<"$(submit_table2 unified "$unified_adapter" "$unified_required" "")"

construction_adapter="$OUTPUT_ROOT/construction_specialist/adapter"
construction_required="$OUTPUT_ROOT/construction_specialist/TRAINING_COMPLETE"
read -r construction_t1g construction_t1s <<<"$(submit_table1 construction_specialist "$construction_adapter" "$construction_required" "$construction_job" "488,490/0")"

editing_adapter="$OUTPUT_ROOT/editing_specialist/adapter"
editing_required="$OUTPUT_ROOT/editing_specialist/TRAINING_COMPLETE"
read -r editing_t2g editing_t2s <<<"$(submit_table2 editing_specialist "$editing_adapter" "$editing_required" "$editing_job")"

collect_job=$(sbatch --parsable --account="$ACCOUNT" --job-name=p29-collect \
  --time=00:30:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$unified_t1s:$unified_t2s:$construction_t1s:$editing_t2s" \
  --kill-on-invalid-dep=yes --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P29_SCRIPT_DIR="$SCRIPT_DIR",P29_OUTPUT_ROOT="$OUTPUT_ROOT" \
  "$SCRIPT_DIR/run_collect.sh")

printf 'construction_train_job=%s\n' "$construction_job"
printf 'editing_train_job=%s\n' "$editing_job"
printf 'unified_table1_jobs=%s,%s\n' "$unified_t1g" "$unified_t1s"
printf 'unified_table2_jobs=%s,%s\n' "$unified_t2g" "$unified_t2s"
printf 'construction_table1_jobs=%s,%s\n' "$construction_t1g" "$construction_t1s"
printf 'editing_table2_jobs=%s,%s\n' "$editing_t2g" "$editing_t2s"
printf 'collect_job=%s\n' "$collect_job"
