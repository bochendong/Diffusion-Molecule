#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P301="$PROJECT/outputs/p30_1_alignment_raw1_rl/seed_30101"
OUT="${P303_OUTPUT_ROOT:-$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301}"
LOG_DIR="$PROJECT/logs/p30_3_joint_multinegative_refinement"
GPU="${P303_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
test -f "$P301/model/balanced_shared_rl/checkpoint-030/adapter/adapter_model.safetensors"

preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p303-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
prepare=$(sbatch --parsable --account=def-hup-ab --job-name=p303-data --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$preflight" --output="$LOG_DIR/prepare-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_prepare.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p303-train --time=02:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$prepare" --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_train.sh")
edit_base=$(sbatch --parsable --account=def-hup-ab --job-name=p303-ebase --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$preflight" --output="$LOG_DIR/edit-baseline-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT",P303_EDIT_TAG=baseline "$SCRIPT_DIR/run_edit_gate.sh")
denovo=$(sbatch --parsable --account=def-hup-ab --job-name=p303-denovo --time=00:20:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$train" --output="$LOG_DIR/denovo-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_denovo_gate.sh")
edit_rl=$(sbatch --parsable --account=def-hup-ab --job-name=p303-edit --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$train" --output="$LOG_DIR/edit-refined-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT",P303_EDIT_TAG=refined "$SCRIPT_DIR/run_edit_gate.sh")
collect=$(sbatch --parsable --account=def-hup-ab --job-name=p303-joint --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$edit_base:$denovo:$edit_rl" --output="$LOG_DIR/collect-%j.log" \
  --export=ALL,P303_SCRIPT_DIR="$SCRIPT_DIR",P303_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_collect.sh")
printf 'preflight_job=%s\nprepare_job=%s\ntrain_job=%s\nediting_baseline_job=%s\ndenovo_gate_job=%s\nediting_refined_job=%s\njoint_collect_job=%s\n' \
  "$preflight" "$prepare" "$train" "$edit_base" "$denovo" "$edit_rl" "$collect"
