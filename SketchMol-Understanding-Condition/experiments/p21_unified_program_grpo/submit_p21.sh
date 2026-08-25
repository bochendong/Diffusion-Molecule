#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p21_unified_program_grpo"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account=def-hup-ab --job-name=p21-prepare --time=00:10:00 --cpus-per-task=4 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export=ALL,P21_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_prepare.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p21-grpo --time=01:15:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/train-%j.log" --export=ALL,P21_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_train.sh")
evaluate=$(sbatch --parsable --account=def-hup-ab --job-name=p21-eval --time=01:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$train" \
  --output="$LOG_DIR/eval-%j.log" --export=ALL,P21_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_eval.sh")
printf 'prepare_job=%s\ntrain_job=%s\neval_job=%s\n' "$prepare" "$train" "$evaluate"
