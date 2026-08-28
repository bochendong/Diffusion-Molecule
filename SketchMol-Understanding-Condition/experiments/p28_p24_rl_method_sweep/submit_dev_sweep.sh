#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P26_DIR="$PROJECT/experiments/p26_decoupled_joint_rl"
INPUT_ADAPTER="${P28_INPUT_ADAPTER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/checkpoint-11000}"
OUT="${P28_OUTPUT_ROOT:-$PROJECT/outputs/p28_p24_rl_method_sweep/checkpoint_11000_seed_28001}"
LOG_DIR="$PROJECT/logs/p28_p24_rl_method_sweep"
mkdir -p "$LOG_DIR"

[[ -f "$INPUT_ADAPTER/adapter_model.safetensors" ]] || {
  echo "ERROR: incomplete P24 checkpoint: $INPUT_ADAPTER" >&2
  exit 2
}

preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p28-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P28_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_preflight.sh")
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p28-c11-base --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --dependency="afterok:$preflight" --output="$LOG_DIR/dev-baseline-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_EVAL_TAG=p24_ckpt11000,P26_EVAL_ADAPTER="$INPUT_ADAPTER",P26_EVAL_REQUIRED="$INPUT_ADAPTER/adapter_model.safetensors" \
  "$P26_DIR/run_gate_eval.sh")

declare -A train_jobs eval_jobs compare_jobs
for variant in vanilla_paired_grpo decoupled_no_pcgrad editing_protected_pcgrad; do
  train_jobs[$variant]=$(sbatch --parsable --account=def-hup-ab --job-name="p28-${variant:0:6}" --time=01:30:00 \
    --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
    --dependency="afterok:$preflight" --output="$LOG_DIR/train-${variant}-%j.log" \
    --export=ALL,P28_SCRIPT_DIR="$SCRIPT_DIR",P28_INPUT_ADAPTER="$INPUT_ADAPTER",P28_OUTPUT_ROOT="$OUT",P28_VARIANT="$variant" \
    "$SCRIPT_DIR/run_variant.sh")
  step20="$OUT/model/$variant/checkpoint-020/adapter"
  eval_jobs[$variant]=$(sbatch --parsable --account=def-hup-ab --job-name="p28-e-${variant:0:4}" --time=00:45:00 \
    --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
    --dependency="afterok:${train_jobs[$variant]}" --output="$LOG_DIR/eval-${variant}-%j.log" \
    --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_EVAL_TAG="$variant",P26_EVAL_ADAPTER="$step20",P26_EVAL_REQUIRED="$OUT/TRAIN_${variant}_COMPLETE" \
    "$P26_DIR/run_gate_eval.sh")
  compare_jobs[$variant]=$(sbatch --parsable --account=def-hup-ab --job-name="p28-c-${variant:0:4}" --time=00:10:00 \
    --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline:${eval_jobs[$variant]}" \
    --output="$LOG_DIR/compare-${variant}-%j.log" --wrap \
    "/home/bdong/.venvs/molscribe_overlay/bin/python '$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py' --baseline '$OUT/gate/dev/p24_ckpt11000/summary.json' --rl '$OUT/gate/dev/$variant/summary.json' --output '$OUT/gate/dev/comparison_${variant}.json'")
done

printf 'preflight_job=%s\nbaseline_job=%s\n' "$preflight" "$baseline"
for variant in vanilla_paired_grpo decoupled_no_pcgrad editing_protected_pcgrad; do
  printf '%s_train=%s\n%s_eval=%s\n%s_compare=%s\n' \
    "$variant" "${train_jobs[$variant]}" \
    "$variant" "${eval_jobs[$variant]}" \
    "$variant" "${compare_jobs[$variant]}"
done
