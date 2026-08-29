#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P26_DIR="$PROJECT/experiments/p26_decoupled_joint_rl"
P25_DIR="$PROJECT/experiments/p25_p23_joint_group_rl"
INPUT_ADAPTER="${P30_INPUT_ADAPTER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/adapter}"
P24_MARKER="${P30_P24_MARKER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/TRAINING_COMPLETE}"
OUT="${P30_OUTPUT_ROOT:-$PROJECT/outputs/p30_balanced_shared_policy_rl/seed_30001}"
LOG_DIR="$PROJECT/logs/p30_balanced_shared_policy_rl"
GPU="${P30_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
test -f "$P24_MARKER"
test -f "$INPUT_ADAPTER/adapter_model.safetensors"

preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p30-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P30_SCRIPT_DIR="$SCRIPT_DIR",P30_OUTPUT_ROOT="$OUT",P30_INPUT_ADAPTER="$INPUT_ADAPTER",P30_P24_MARKER="$P24_MARKER" \
  "$SCRIPT_DIR/run_preflight.sh")

train=$(sbatch --parsable --account=def-hup-ab --job-name=p30-balanced-rl --time=03:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P30_SCRIPT_DIR="$SCRIPT_DIR",P30_OUTPUT_ROOT="$OUT",P30_INPUT_ADAPTER="$INPUT_ADAPTER",P30_P24_MARKER="$P24_MARKER" \
  "$SCRIPT_DIR/run_train.sh")

baseline_dev=$(sbatch --parsable --account=def-hup-ab --job-name=p30-base-dev --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/baseline-dev-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_GATE_SPLIT=dev,P26_EVAL_TAG=baseline,P26_EVAL_ADAPTER="$INPUT_ADAPTER",P26_EVAL_REQUIRED="$P24_MARKER" \
  "$P26_DIR/run_gate_eval.sh")

baseline_final=$(sbatch --parsable --account=def-hup-ab --job-name=p30-base-final --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/baseline-final-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_GATE_SPLIT=final,P26_EVAL_TAG=baseline,P26_EVAL_ADAPTER="$INPUT_ADAPTER",P26_EVAL_REQUIRED="$P24_MARKER" \
  "$P26_DIR/run_gate_eval.sh")

eval_jobs=()
compare_jobs=()
for step in 010 020 030 040 050 060; do
  adapter="$OUT/model/balanced_shared_rl/checkpoint-$step/adapter"
  marker="$OUT/model/balanced_shared_rl/checkpoint-$step/CHECKPOINT_COMPLETE"
  eval_job=$(sbatch --parsable --account=def-hup-ab --job-name="p30-e${step}" --time=00:45:00 \
    --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$train" \
    --output="$LOG_DIR/dev-step${step}-%j.log" \
    --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_GATE_SPLIT=dev,P26_EVAL_TAG="step${step}",P26_EVAL_ADAPTER="$adapter",P26_EVAL_REQUIRED="$marker" \
    "$P26_DIR/run_gate_eval.sh")
  compare_job=$(sbatch --parsable --account=def-hup-ab --job-name="p30-c${step}" --time=00:10:00 \
    --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline_dev:$eval_job" \
    --output="$LOG_DIR/compare-step${step}-%j.log" --wrap \
    "/home/bdong/.venvs/molscribe_overlay/bin/python '$P25_DIR/compare_gate.py' --baseline '$OUT/gate/dev/baseline/summary.json' --rl '$OUT/gate/dev/step${step}/summary.json' --output '$OUT/gate/dev/comparison_step${step}.json'")
  eval_jobs+=("$eval_job")
  compare_jobs+=("$compare_job")
done

compare_dependency=$(IFS=:; echo "${compare_jobs[*]}")
select_job=$(sbatch --parsable --account=def-hup-ab --job-name=p30-select --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$compare_dependency" \
  --output="$LOG_DIR/select-%j.log" \
  --export=ALL,P30_SCRIPT_DIR="$SCRIPT_DIR",P30_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_select.sh")

final_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p30-final-rl --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$select_job" \
  --output="$LOG_DIR/final-selected-%j.log" \
  --export=ALL,P30_SCRIPT_DIR="$SCRIPT_DIR",P30_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_selected_final_eval.sh")

final_compare=$(sbatch --parsable --account=def-hup-ab --job-name=p30-final-cmp --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline_final:$final_eval" \
  --output="$LOG_DIR/final-compare-%j.log" --wrap \
  "/home/bdong/.venvs/molscribe_overlay/bin/python '$P25_DIR/compare_gate.py' --baseline '$OUT/gate/final/baseline/summary.json' --rl '$OUT/gate/final/selected/summary.json' --output '$OUT/gate/final/comparison.json'")

collect=$(sbatch --parsable --account=def-hup-ab --job-name=p30-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$final_compare" \
  --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P30_SCRIPT_DIR="$SCRIPT_DIR",P30_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")

printf 'preflight_job=%s\ntrain_job=%s\nbaseline_dev_job=%s\nbaseline_final_job=%s\n' \
  "$preflight" "$train" "$baseline_dev" "$baseline_final"
printf 'dev_eval_jobs=%s\ndev_compare_jobs=%s\nselection_job=%s\nfinal_eval_job=%s\nfinal_compare_job=%s\ncollect_job=%s\n' \
  "$(IFS=,; echo "${eval_jobs[*]}")" "$(IFS=,; echo "${compare_jobs[*]}")" \
  "$select_job" "$final_eval" "$final_compare" "$collect"
