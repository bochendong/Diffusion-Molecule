#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; LOG="$PROJECT_DIR/logs/p8_1_12_verified_success_distill"; mkdir -p "$LOG"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
common=(--parsable --account="${P8112_ACCOUNT:-rrg-hup}" --time="${P8112_TIME:-00:15:00}" --mem=40G --cpus-per-task=8 --gpus="${P8112_GPU:-h100:1}" --export=ALL)
pre="$(sbatch "${common[@]}" --job-name=p8112-pre-s7 --output="$LOG/pre-%j.log" --wrap="bash '$SCRIPT_DIR/run_precompute.sh'")"; pre="${pre%%;*}"
r1="$(sbatch "${common[@]}" --dependency="afterany:$pre" --job-name=p8112-r1-s7 --output="$LOG/r1-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r1")"; r1="${r1%%;*}"
r2="$(sbatch "${common[@]}" --dependency="afterany:$r1" --job-name=p8112-r2-s7 --output="$LOG/r2-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r2")"; r2="${r2%%;*}"
printf 'P8.1.12 PRE=%s\nP8.1.12 R1=%s dependency=afterany:%s\nP8.1.12 R2=%s dependency=afterany:%s\n' "$pre" "$r1" "$pre" "$r2" "$r1"
