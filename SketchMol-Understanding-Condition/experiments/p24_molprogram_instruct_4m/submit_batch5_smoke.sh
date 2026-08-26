#!/usr/bin/env bash
set -euo pipefail

dependency="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
GRES="${P24_GRES:-gpu:h100:1}"
SMOKE_ROOT="${P24_BATCH5_SMOKE_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/batch5_smoke_seed24003}"
SMOKE_STEPS="${P24_BATCH5_SMOKE_STEPS:-20}"
SMOKE_WALLTIME="${P24_BATCH5_SMOKE_WALLTIME:-00:30:00}"
mkdir -p "$LOG_DIR"
dep_args=()
[[ -n "$dependency" ]] && dep_args+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
job=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-b5-smoke --time="$SMOKE_WALLTIME" \
  --cpus-per-task=6 --mem=64G --gres="$GRES" "${dep_args[@]}" \
  --output="$LOG_DIR/batch5-smoke-%j.log" \
  --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR",P24_TRAIN_MODE=full,P24_OUTPUT_ROOT="$SMOKE_ROOT",P24_MAX_STEPS="$SMOKE_STEPS",P24_BATCH_SIZE=5,P24_GRADIENT_ACCUMULATION=13,P24_SAVE_STEPS=1000 \
  "$SCRIPT_DIR/run_train.sh")
printf 'batch5_smoke_job=%s\n' "$job"
