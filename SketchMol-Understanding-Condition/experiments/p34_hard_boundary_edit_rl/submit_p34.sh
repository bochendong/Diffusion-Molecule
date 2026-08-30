#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P34_OUTPUT_ROOT:-$PROJECT/outputs/p34_hard_boundary_edit_rl/seed_34001}"
LOG_DIR="$PROJECT/logs/p34_hard_boundary_edit_rl"
ACCOUNT="${P34_ACCOUNT:-def-hup-ab}"
GPU="${P34_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=p34-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P34_SCRIPT_DIR="$SCRIPT_DIR",P34_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=p34-base --time=00:05:00 \
  --cpus-per-task=1 --mem=2G --dependency="afterok:$preflight" --output="$LOG_DIR/prepare-%j.log" \
  --export=ALL,P34_SCRIPT_DIR="$SCRIPT_DIR",P34_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_prepare.sh")
train=$(sbatch --parsable --account="$ACCOUNT" --job-name=p34-train --time=01:00:00 \
  --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$prepare" \
  --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P34_SCRIPT_DIR="$SCRIPT_DIR",P34_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_train.sh")
eval_jobs=()
for step in 005 010 020; do
  job=$(sbatch --parsable --account="$ACCOUNT" --job-name="p34-e$step" --time=00:30:00 \
    --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$train" \
    --output="$LOG_DIR/eval-$step-%j.log" \
    --export=ALL,P34_SCRIPT_DIR="$SCRIPT_DIR",P34_OUTPUT_ROOT="$OUT",P34_STEP="$step" "$SCRIPT_DIR/run_eval.sh")
  eval_jobs+=("$job")
done
eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=p34-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$eval_dependency:$prepare" \
  --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P34_SCRIPT_DIR="$SCRIPT_DIR",P34_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\nprepare_job=%s\ntrain_job=%s\neval_jobs=%s\ncollect_job=%s\n' \
  "$preflight" "$prepare" "$train" "${eval_jobs[*]}" "$collect"
