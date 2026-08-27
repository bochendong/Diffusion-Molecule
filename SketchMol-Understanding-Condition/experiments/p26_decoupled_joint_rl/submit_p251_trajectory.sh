#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p26_decoupled_joint_rl"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
OUT="$PROJECT/outputs/p26_decoupled_joint_rl/seed_26001"
mkdir -p "$LOG_DIR"
for step in 010 020; do
  adapter="$P251/model/p25_1/checkpoint-$step/adapter"
  [[ -f "$adapter/adapter_model.safetensors" ]] || { echo "ERROR: missing checkpoint $adapter" >&2; exit 2; }
  eval_job=$(sbatch --parsable --account=def-hup-ab --job-name="p26-p251-$step" --time=00:45:00 \
    --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
    --output="$LOG_DIR/p251-step${step}-%j.log" \
    --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR",P26_EVAL_TAG="p251_step${step}",P26_EVAL_ADAPTER="$adapter",P26_EVAL_REQUIRED="$P251/model/p25_1/checkpoint-$step/CHECKPOINT_COMPLETE" \
    "$SCRIPT_DIR/run_gate_eval.sh")
  compare_job=$(sbatch --parsable --account=def-hup-ab --job-name="p26-cmp-$step" --time=00:10:00 \
    --cpus-per-task=2 --mem=8G --dependency="afterok:$eval_job" \
    --output="$LOG_DIR/p251-compare${step}-%j.log" --wrap \
    "/home/bdong/.venvs/molscribe_overlay/bin/python '$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py' --baseline '$P251/gate/dev/baseline/summary.json' --rl '$OUT/gate/dev/p251_step${step}/summary.json' --output '$OUT/gate/dev/p251_step${step}/comparison.json'")
  printf 'p251_step%s_eval_job=%s\np251_step%s_compare_job=%s\n' "$step" "$eval_job" "$step" "$compare_job"
done
