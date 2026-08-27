#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p26_decoupled_joint_rl"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
OUT="$PROJECT/outputs/p26_decoupled_joint_rl/seed_26001"
mkdir -p "$LOG_DIR"
preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p26-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_preflight.sh")
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p26-dev-base --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$preflight" --output="$LOG_DIR/dev-baseline-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR",P26_EVAL_TAG=baseline,P26_EVAL_ADAPTER="$P23/model/stage1_v2/adapter",P26_EVAL_REQUIRED="$P23/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$SCRIPT_DIR/run_gate_eval.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p26-joint-rl --time=01:30:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$preflight" --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_train.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p26-dev-rl --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$train" --output="$LOG_DIR/dev-rl-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR",P26_EVAL_TAG=p26,P26_EVAL_ADAPTER="$OUT/model/p26/adapter",P26_EVAL_REQUIRED="$OUT/TRAIN_COMPLETE" \
  "$SCRIPT_DIR/run_gate_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p26-dev-cmp --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline:$rl_eval" \
  --output="$LOG_DIR/dev-compare-%j.log" --wrap \
  "/home/bdong/.venvs/molscribe_overlay/bin/python '$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py' --baseline '$OUT/gate/dev/baseline/summary.json' --rl '$OUT/gate/dev/p26/summary.json' --output '$OUT/gate/dev/comparison.json'")
printf 'preflight_job=%s\ndev_baseline_job=%s\ntrain_job=%s\ndev_rl_job=%s\ndev_compare_job=%s\n' \
  "$preflight" "$baseline" "$train" "$rl_eval" "$compare"
