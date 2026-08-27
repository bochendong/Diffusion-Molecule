#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p25_1_p23_mode_paired_grpo"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account=def-hup-ab --job-name=p251-prepare --time=00:10:00 --cpus-per-task=4 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_prepare.sh")
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p251-dev-base --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/dev-baseline-%j.log" --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR",P251_GATE_SPLIT=dev,P251_EVAL_METHOD=baseline "$SCRIPT_DIR/run_eval.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p251-paired-rl --time=01:00:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/train-%j.log" --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_train.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p251-dev-rl --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$train" \
  --output="$LOG_DIR/dev-rl-%j.log" --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR",P251_GATE_SPLIT=dev,P251_EVAL_METHOD=rl "$SCRIPT_DIR/run_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p251-dev-cmp --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$baseline:$rl_eval" --output="$LOG_DIR/dev-compare-%j.log" \
  --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR",P251_GATE_SPLIT=dev "$SCRIPT_DIR/run_compare.sh")
printf 'prepare_job=%s\ndev_baseline_job=%s\ntrain_job=%s\ndev_rl_job=%s\ndev_compare_job=%s\n' \
  "$prepare" "$baseline" "$train" "$rl_eval" "$compare"
