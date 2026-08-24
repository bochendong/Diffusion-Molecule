#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; LOG="$PROJECT/logs/p8_1_13_verified_counterfactual_dpo"; mkdir -p "$LOG"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
common=(--parsable --account="${P8113_ACCOUNT:-rrg-hup}" --time="${P8113_TIME:-00:15:00}" --mem=40G --cpus-per-task=8 --gpus="${P8113_GPU:-h100:1}" --export=ALL)
UP="$PROJECT/outputs/p8_1_12_verified_success_distill/shared/seed_7/PRECOMPUTE_COMPLETE"; dep=()
if [[ ! -s "$UP" ]]; then [[ -n "${P8112_PRE_JOB_ID:-}" ]] || { echo "Set P8112_PRE_JOB_ID when shared P8.1.12 PRE is absent" >&2; exit 44; }; dep=(--dependency="afterany:${P8112_PRE_JOB_ID}"); fi
pre="$(sbatch "${common[@]}" "${dep[@]}" --job-name=p8113-pre-s7 --output="$LOG/pre-%j.log" --wrap="bash '$HERE/run_precompute.sh'")"; pre="${pre%%;*}"
r1="$(sbatch "${common[@]}" --dependency="afterany:$pre" --job-name=p8113-r1-s7 --output="$LOG/r1-%j.log" --wrap="bash '$HERE/run_round.sh' r1")"; r1="${r1%%;*}"
r2="$(sbatch "${common[@]}" --dependency="afterany:$r1" --job-name=p8113-r2-s7 --output="$LOG/r2-%j.log" --wrap="bash '$HERE/run_round.sh' r2")"; r2="${r2%%;*}"
printf 'P8.1.13 PRE=%s\nP8.1.13 R1=%s dependency=afterany:%s\nP8.1.13 R2=%s dependency=afterany:%s\n' "$pre" "$r1" "$pre" "$r2" "$r1"
