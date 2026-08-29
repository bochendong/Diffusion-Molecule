#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P301_OUTPUT_ROOT:-$PROJECT/outputs/p30_1_alignment_raw1_rl/seed_30101}"
LOG_DIR="$PROJECT/logs/p30_1_alignment_raw1_rl"
GPU="${P301_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
test -f "$OUT/TRAIN_COMPLETE"
test -f "$OUT/SMALL_GATE_DATA_COMPLETE"
test -f "$OUT/small_gate/rl/result.json"
mkdir -p "$LOG_DIR"

eval_jobs=()
for step in 010 020; do
  adapter="$OUT/model/balanced_shared_rl/checkpoint-$step/adapter"
  test -f "$adapter/adapter_model.safetensors"
  job=$(sbatch --parsable --account=def-hup-ab --job-name="p301-s${step}" --time=00:20:00 \
    --cpus-per-task=4 --mem=48G --gres="$GPU" \
    --output="$LOG_DIR/raw1-step${step}-%j.log" \
    --export=ALL,P301_SCRIPT_DIR="$SCRIPT_DIR",P301_OUTPUT_ROOT="$OUT",P301_EVAL_TAG="step${step}",P301_EVAL_ADAPTER="$adapter" \
    "$SCRIPT_DIR/run_small_gate.sh")
  eval_jobs+=("$job")
done

dependency=$(IFS=:; echo "${eval_jobs[*]}")
select=$(sbatch --parsable --account=def-hup-ab --job-name=p301-select --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$dependency" \
  --output="$LOG_DIR/select-screens-%j.log" \
  --export=ALL,P301_SCRIPT_DIR="$SCRIPT_DIR",P301_OUTPUT_ROOT="$OUT" \
  "$SCRIPT_DIR/run_select_screens.sh")
printf 'checkpoint_screen_jobs=%s\nselection_job=%s\n' "$(IFS=,; echo "${eval_jobs[*]}")" "$select"

