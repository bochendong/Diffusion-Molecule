#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: submit_train.sh gate|full [dependency_job]}"
dependency="${2:-}"
[[ "$mode" == gate || "$mode" == full ]] || { echo "mode must be gate or full" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
GRES="${P24_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"
dep_args=()
[[ -n "$dependency" ]] && dep_args+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
if [[ "$mode" == gate ]]; then
  walltime="04:00:00"
  max_steps=500
  batch_size=1
  accumulation=26
  job_name=p24-gate13k
else
  # The corrected gate measured about 4.3 seconds per optimizer step at
  # accumulation 26. Full accumulation 65 therefore needs roughly 50 hours.
  walltime="3-00:00:00"
  max_steps=16283
  batch_size=1
  accumulation=65
  job_name=p24-full
fi
job=$(sbatch --parsable --account="$ACCOUNT" --job-name="$job_name" --time="$walltime" \
  --cpus-per-task=6 --mem=64G --gres="$GRES" "${dep_args[@]}" \
  --output="$LOG_DIR/train-$mode-%j.log" \
  --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR",P24_TRAIN_MODE="$mode",P24_MAX_STEPS="$max_steps",P24_BATCH_SIZE="$batch_size",P24_GRADIENT_ACCUMULATION="$accumulation" \
  "$SCRIPT_DIR/run_train.sh")
printf '%s_job=%s\n' "$mode" "$job"
