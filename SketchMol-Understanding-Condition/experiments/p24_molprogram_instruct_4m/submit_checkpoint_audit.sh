#!/usr/bin/env bash
set -euo pipefail

step="${1:?usage: submit_checkpoint_audit.sh CHECKPOINT_STEP}"
[[ "$step" =~ ^[0-9]+$ ]] || { echo "checkpoint step must be numeric" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
ADAPTER="$P24_OUT/full/checkpoint-$step"
REQUIRED="$ADAPTER/adapter_model.safetensors"
AUDIT="$P24_OUT/checkpoint_audits/checkpoint-$step"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m/checkpoint-$step"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
GRES="${P24_AUDIT_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
LABEL="p24_checkpoint_$step"
test -f "$REQUIRED"
mkdir -p "$LOG_DIR"

common="ALL,P24_SCRIPT_DIR=$SCRIPT_DIR,P24_EVAL_ADAPTER=$ADAPTER,P24_EVAL_REQUIRED=$REQUIRED,P24_EVAL_LABEL=$LABEL"
t1_out="$AUDIT/eval_table1"
t2_out="$AUDIT/eval_table2"

t1_generation=$(sbatch --parsable --account="$ACCOUNT" --job-name="p24-c${step}-t1g" \
  --time=06:00:00 --cpus-per-task=4 --mem=48G --gres="$GRES" \
  --output="$LOG_DIR/table1-generate-%j.log" \
  --export="$common,P24_TABLE1_OUT=$t1_out" "$SCRIPT_DIR/run_table1_generate.sh")
t1_scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name="p24-c${step}-t1s" \
  --time=01:00:00 --cpus-per-task=4 --mem=24G \
  --dependency="afterok:$t1_generation" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/table1-score-%j.log" \
  --export="ALL,P24_SCRIPT_DIR=$SCRIPT_DIR,P24_TABLE1_OUT=$t1_out,P24_TABLE1_MODEL_NAME=P24-checkpoint-$step,P24_TABLE1_PROTOCOL=p24_checkpoint_${step}_denovo_table1_best40_v1" \
  "$SCRIPT_DIR/run_table1_finalize.sh")

t2_generation=$(sbatch --parsable --account="$ACCOUNT" --job-name="p24-c${step}-t2g" \
  --time=03:00:00 --cpus-per-task=4 --mem=48G --gres="$GRES" \
  --output="$LOG_DIR/table2-generate-%j.log" \
  --export="$common,P24_TABLE2_OUT=$t2_out" "$SCRIPT_DIR/run_table2_generate.sh")
t2_scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name="p24-c${step}-t2s" \
  --time=01:00:00 --cpus-per-task=4 --mem=24G \
  --dependency="afterok:$t2_generation" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/table2-score-%j.log" \
  --export="ALL,P24_SCRIPT_DIR=$SCRIPT_DIR,P24_TABLE2_OUT=$t2_out,P24_TABLE2_MODEL_NAME=P24-checkpoint-$step-sampled-once" \
  "$SCRIPT_DIR/run_table2_score.sh")

printf 'table1_generation_job=%s\ntable1_scoring_job=%s\n' "$t1_generation" "$t1_scoring"
printf 'table2_generation_job=%s\ntable2_scoring_job=%s\n' "$t2_generation" "$t2_scoring"
