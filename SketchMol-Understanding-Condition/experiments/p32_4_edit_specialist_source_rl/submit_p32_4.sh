#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P324_OUTPUT_ROOT:-$PROJECT/outputs/p32_4_edit_specialist_source_rl/seed_32401}"
LOG_DIR="$PROJECT/logs/p32_4_edit_specialist_source_rl"
ACCOUNT="${P324_ACCOUNT:-def-hup-ab}"
GPU="${P324_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"

preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=p324-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P324_SCRIPT_DIR="$SCRIPT_DIR",P324_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
baseline=$(sbatch --parsable --account="$ACCOUNT" --job-name=p324-e000 --time=01:00:00 \
  --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/eval-000-%j.log" \
  --export=ALL,P324_SCRIPT_DIR="$SCRIPT_DIR",P324_OUTPUT_ROOT="$OUT",P324_STEP=000 "$SCRIPT_DIR/run_eval.sh")
train=$(sbatch --parsable --account="$ACCOUNT" --job-name=p324-train --time=04:00:00 \
  --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P324_SCRIPT_DIR="$SCRIPT_DIR",P324_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_train.sh")

eval_jobs=()
for step in 005 010 020; do
  job=$(sbatch --parsable --account="$ACCOUNT" --job-name="p324-e$step" --time=01:00:00 \
    --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$train" \
    --output="$LOG_DIR/eval-$step-%j.log" \
    --export=ALL,P324_SCRIPT_DIR="$SCRIPT_DIR",P324_OUTPUT_ROOT="$OUT",P324_STEP="$step" "$SCRIPT_DIR/run_eval.sh")
  eval_jobs+=("$job")
done
eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=p324-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline:$eval_dependency" \
  --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P324_SCRIPT_DIR="$SCRIPT_DIR",P324_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\nbaseline_eval_job=%s\ntrain_job=%s\neval_jobs=%s\ncollect_job=%s\n' \
  "$preflight" "$baseline" "$train" "${eval_jobs[*]}" "$collect"
