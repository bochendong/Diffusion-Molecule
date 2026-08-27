#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p25_p23_joint_group_rl"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account=def-hup-ab --job-name=p25-prepare --time=00:10:00 --cpus-per-task=4 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_prepare.sh")
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p25-base-gate --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/baseline-%j.log" --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR",P25_EVAL_METHOD=baseline "$SCRIPT_DIR/run_eval.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p25-p23-grpo --time=03:15:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/train-%j.log" --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_train.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p25-rl-gate --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$train" \
  --output="$LOG_DIR/rl-eval-%j.log" --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR",P25_EVAL_METHOD=rl "$SCRIPT_DIR/run_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p25-compare --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$baseline:$rl_eval" --output="$LOG_DIR/compare-%j.log" \
  --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_compare.sh")
printf 'prepare_job=%s\nbaseline_gate_job=%s\ntrain_job=%s\nrl_gate_job=%s\ncompare_job=%s\n' \
  "$prepare" "$baseline" "$train" "$rl_eval" "$compare"
