#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
INPUT_ADAPTER="${P301_INPUT_ADAPTER:-$P24/alignment_refresh/model/adapter}"
P24_MARKER="${P301_P24_MARKER:-$P24/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE}"
OUT="${P301_OUTPUT_ROOT:-$PROJECT/outputs/p30_1_alignment_raw1_rl/seed_30101}"
LOG_DIR="$PROJECT/logs/p30_1_alignment_raw1_rl"
GPU="${P301_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
test -f "$P24_MARKER"
test -f "$INPUT_ADAPTER/adapter_model.safetensors"

preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p301-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P301_SCRIPT_DIR="$SCRIPT_DIR",P301_OUTPUT_ROOT="$OUT",P301_INPUT_ADAPTER="$INPUT_ADAPTER",P301_P24_MARKER="$P24_MARKER" \
  "$SCRIPT_DIR/run_preflight.sh")

build=$(sbatch --parsable --account=def-hup-ab --job-name=p301-build --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$preflight" \
  --output="$LOG_DIR/build-%j.log" \
  --export=ALL,P301_SCRIPT_DIR="$SCRIPT_DIR",P301_OUTPUT_ROOT="$OUT" \
  "$SCRIPT_DIR/run_build_gate.sh")

train=$(sbatch --parsable --account=def-hup-ab --job-name=p301-rl --time=03:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$preflight" \
  --output="$LOG_DIR/train-%j.log" \
  --export=ALL,P301_SCRIPT_DIR="$SCRIPT_DIR",P301_OUTPUT_ROOT="$OUT",P301_INPUT_ADAPTER="$INPUT_ADAPTER",P301_P24_MARKER="$P24_MARKER" \
  "$SCRIPT_DIR/run_train.sh")

gate=$(sbatch --parsable --account=def-hup-ab --job-name=p301-raw1 --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$train:$build" \
  --output="$LOG_DIR/raw1-%j.log" \
  --export=ALL,P301_SCRIPT_DIR="$SCRIPT_DIR",P301_OUTPUT_ROOT="$OUT" \
  "$SCRIPT_DIR/run_small_gate.sh")

printf 'preflight_job=%s\nbuild_job=%s\ntrain_job=%s\nsmall_raw1_gate_job=%s\n' "$preflight" "$build" "$train" "$gate"

