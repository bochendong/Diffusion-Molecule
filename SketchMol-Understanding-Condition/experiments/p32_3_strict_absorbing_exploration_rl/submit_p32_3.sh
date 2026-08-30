#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P323_OUTPUT_ROOT:-$PROJECT/outputs/p32_3_strict_absorbing_exploration_rl/seed_32301}"
LOG_DIR="$PROJECT/logs/p32_3_strict_absorbing_exploration_rl"
ACCOUNT="${P323_ACCOUNT:-def-hup-ab}"
GPU="${P323_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"

preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=p323-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P323_SCRIPT_DIR="$SCRIPT_DIR",P323_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
audit=$(sbatch --parsable --account="$ACCOUNT" --job-name=p323-audit --time=01:00:00 \
  --cpus-per-task=4 --mem=16G --dependency="afterok:$preflight" \
  --output="$LOG_DIR/audit-%j.log" \
  --export=ALL,P323_SCRIPT_DIR="$SCRIPT_DIR",P323_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_support_audit.sh")
train=$(sbatch --parsable --account="$ACCOUNT" --job-name=p323-train --time=03:00:00 \
  --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$audit" \
  --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P323_SCRIPT_DIR="$SCRIPT_DIR",P323_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_train.sh")

eval_jobs=()
for step in 000 005 010 020; do
  dependency="$preflight"
  [[ "$step" == "000" ]] || dependency="$train"
  job=$(sbatch --parsable --account="$ACCOUNT" --job-name="p323-e$step" --time=01:00:00 \
    --cpus-per-task=4 --mem=40G --gres="$GPU" --dependency="afterok:$dependency" \
    --output="$LOG_DIR/eval-$step-%j.log" \
    --export=ALL,P323_SCRIPT_DIR="$SCRIPT_DIR",P323_OUTPUT_ROOT="$OUT",P323_STEP="$step" "$SCRIPT_DIR/run_eval.sh")
  eval_jobs+=("$job")
done
eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=p323-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$eval_dependency" \
  --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P323_SCRIPT_DIR="$SCRIPT_DIR",P323_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\naudit_job=%s\ntrain_job=%s\neval_jobs=%s\ncollect_job=%s\n' \
  "$preflight" "$audit" "$train" "${eval_jobs[*]}" "$collect"
