#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P311_OUTPUT_ROOT:-$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101}"
LOG_DIR="$PROJECT/logs/p31_1_frontier_online_rloo"
GPU="${P311_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p311-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
build=$(sbatch --parsable --account=def-hup-ab --job-name=p311-data --time=00:15:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$preflight" --output="$LOG_DIR/build-%j.log" \
  --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_build_gate.sh")
train_jobs=()
for mode in de_novo edit; do
  job=$(sbatch --parsable --account=def-hup-ab --job-name="p311-${mode}" --time=08:00:00 \
    --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$build" \
    --output="$LOG_DIR/train-${mode}-%j.log" \
    --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT",P311_MODE="$mode" "$SCRIPT_DIR/run_train_mode.sh")
  train_jobs+=("$job")
done
train_dependency=$(IFS=:; echo "${train_jobs[*]}")
eval_jobs=()
for step in 025 050 100; do
  job=$(sbatch --parsable --account=def-hup-ab --job-name="p311-e${step}" --time=03:00:00 \
    --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$train_dependency" \
    --output="$LOG_DIR/eval-${step}-%j.log" \
    --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT",P311_STEP="$step" "$SCRIPT_DIR/run_eval_checkpoint.sh")
  eval_jobs+=("$job")
done
eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account=def-hup-ab --job-name=p311-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$eval_dependency" --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\nbuild_job=%s\ntrain_jobs=%s\neval_jobs=%s\ncollect_job=%s\n' \
  "$preflight" "$build" "${train_jobs[*]}" "${eval_jobs[*]}" "$collect"
