#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; LOG="$PROJECT_DIR/logs/p8_1_11_transduction_group_rl"; mkdir -p "$LOG"; export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
common=(--parsable --account="${P8111_ACCOUNT:-rrg-hup}" --time="${P8111_TIME:-00:15:00}" --mem=48G --cpus-per-task=8 --gpus="${P8111_GPU:-h100:1}" --export=ALL)
r1="$(sbatch "${common[@]}" --job-name=p8.1.11-r1-s7 --output="$LOG/r1-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r1")"; r1="${r1%%;*}"
r2="$(sbatch "${common[@]}" --dependency="afterany:$r1" --job-name=p8.1.11-r2-s7 --output="$LOG/r2-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r2")"; r2="${r2%%;*}"
printf 'P8.1.11 R1=%s\nP8.1.11 R2=%s dependency=afterany:%s\n' "$r1" "$r2" "$r1"
