#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P33_OUTPUT_ROOT:-$PROJECT/outputs/p33_joint_vs_separate_pilot/seed_33001}"
LOG_DIR="$PROJECT/logs/p33_joint_vs_separate_pilot"
ACCOUNT="${P33_ACCOUNT:-def-hup-ab}"
GPU="${P33_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=p33-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P33_SCRIPT_DIR="$SCRIPT_DIR",P33_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=p33-data --time=00:15:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$preflight" --output="$LOG_DIR/prepare-%j.log" \
  --export=ALL,P33_SCRIPT_DIR="$SCRIPT_DIR",P33_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_prepare.sh")
train_jobs=()
eval_jobs=()
for arm in joint denovo edit; do
  train=$(sbatch --parsable --account="$ACCOUNT" --job-name="p33-t-$arm" --time=03:00:00 \
    --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$prepare" \
    --output="$LOG_DIR/train-$arm-%j.log" \
    --export=ALL,P33_SCRIPT_DIR="$SCRIPT_DIR",P33_OUTPUT_ROOT="$OUT",P33_ARM="$arm" "$SCRIPT_DIR/run_train.sh")
  eval=$(sbatch --parsable --account="$ACCOUNT" --job-name="p33-e-$arm" --time=01:00:00 \
    --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$train" \
    --output="$LOG_DIR/eval-$arm-%j.log" \
    --export=ALL,P33_SCRIPT_DIR="$SCRIPT_DIR",P33_OUTPUT_ROOT="$OUT",P33_ARM="$arm" "$SCRIPT_DIR/run_eval.sh")
  train_jobs+=("$train")
  eval_jobs+=("$eval")
done
eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=p33-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$eval_dependency" \
  --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P33_SCRIPT_DIR="$SCRIPT_DIR",P33_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\nprepare_job=%s\ntrain_jobs=%s\neval_jobs=%s\ncollect_job=%s\n' \
  "$preflight" "$prepare" "${train_jobs[*]}" "${eval_jobs[*]}" "$collect"
