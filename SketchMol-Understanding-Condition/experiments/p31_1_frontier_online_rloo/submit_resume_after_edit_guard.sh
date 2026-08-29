#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P311_OUTPUT_ROOT:-$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101}"
LOG_DIR="$PROJECT/logs/p31_1_frontier_online_rloo"
GPU="${P311_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
test -f "$OUT/model/de_novo/TRAIN_COMPLETE"
test -f "$OUT/model/edit/checkpoint-050/CHECKPOINT_COMPLETE"
test -f "$OUT/GATE_DATA_COMPLETE"
mkdir -p "$LOG_DIR"
preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p311r-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/resume-preflight-%j.log" \
  --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
edit=$(sbatch --parsable --account=def-hup-ab --job-name=p311r-edit --time=08:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/resume-edit-%j.log" \
  --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT",P311_MODE=edit "$SCRIPT_DIR/run_train_mode.sh")
eval_jobs=()
for step in 025 050 100; do
  job=$(sbatch --parsable --account=def-hup-ab --job-name="p311r-e${step}" --time=03:00:00 \
    --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$edit" \
    --output="$LOG_DIR/resume-eval-${step}-%j.log" \
    --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT",P311_STEP="$step" "$SCRIPT_DIR/run_eval_checkpoint.sh")
  eval_jobs+=("$job")
done
eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account=def-hup-ab --job-name=p311r-collect --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$eval_dependency" --output="$LOG_DIR/resume-collect-%j.log" \
  --export=ALL,P311_SCRIPT_DIR="$SCRIPT_DIR",P311_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\nedit_resume_job=%s\neval_jobs=%s\ncollect_job=%s\n' \
  "$preflight" "$edit" "${eval_jobs[*]}" "$collect"
