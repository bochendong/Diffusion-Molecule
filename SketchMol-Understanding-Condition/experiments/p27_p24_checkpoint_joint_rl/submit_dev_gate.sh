#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P26_DIR="$PROJECT/experiments/p26_decoupled_joint_rl"
P24_CHECKPOINT="${P27_INPUT_ADAPTER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/checkpoint-7500}"
OUT="${P27_OUTPUT_ROOT:-$PROJECT/outputs/p27_p24_checkpoint_joint_rl/checkpoint_7500_seed_26001}"
LOG_DIR="$PROJECT/logs/p27_p24_checkpoint_joint_rl"
RL_ADAPTER="$OUT/model/p24_ckpt7500_rl/adapter"
mkdir -p "$LOG_DIR"

[[ -f "$P24_CHECKPOINT/adapter_model.safetensors" ]] || {
  echo "ERROR: incomplete P24 checkpoint: $P24_CHECKPOINT" >&2
  exit 2
}

preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p27-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR" "$P26_DIR/run_preflight.sh")
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p27-c75-base --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$preflight" --output="$LOG_DIR/dev-baseline-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_EVAL_TAG=p24_ckpt7500,P26_EVAL_ADAPTER="$P24_CHECKPOINT",P26_EVAL_REQUIRED="$P24_CHECKPOINT/adapter_model.safetensors" \
  "$P26_DIR/run_gate_eval.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p27-c75-rl --time=01:30:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$preflight" --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P27_INPUT_ADAPTER="$P24_CHECKPOINT",P27_OUTPUT_ROOT="$OUT" \
  "$SCRIPT_DIR/run_train.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p27-c75-eval --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$train" --output="$LOG_DIR/dev-rl-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_EVAL_TAG=p24_ckpt7500_rl,P26_EVAL_ADAPTER="$RL_ADAPTER",P26_EVAL_REQUIRED="$OUT/TRAIN_COMPLETE" \
  "$P26_DIR/run_gate_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p27-c75-cmp --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline:$rl_eval" \
  --output="$LOG_DIR/dev-compare-%j.log" --wrap \
  "/home/bdong/.venvs/molscribe_overlay/bin/python '$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py' --baseline '$OUT/gate/dev/p24_ckpt7500/summary.json' --rl '$OUT/gate/dev/p24_ckpt7500_rl/summary.json' --output '$OUT/gate/dev/comparison.json'")

printf 'preflight_job=%s\ndev_baseline_job=%s\ntrain_job=%s\ndev_rl_job=%s\ndev_compare_job=%s\n' \
  "$preflight" "$baseline" "$train" "$rl_eval" "$compare"
