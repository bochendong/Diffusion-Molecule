#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; LOG="$PROJECT_DIR/logs/p8_1_9_transaction_outcome_distill"; mkdir -p "$LOG"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
common=(--parsable --account="${P819_ACCOUNT:-rrg-hup}" --time="${P819_TIME:-00:15:00}" --mem=40G --cpus-per-task=8 --gpus="${P819_GPU:-h100:1}" --export=ALL)
r1="$(sbatch "${common[@]}" --job-name=p819-r1-s7 --output="$LOG/r1-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r1")"; r1="${r1%%;*}"
r2="$(sbatch "${common[@]}" --dependency="afterany:$r1" --job-name=p819-r2-s7 --output="$LOG/r2-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r2")"; r2="${r2%%;*}"
printf 'P8.1.9 R1=%s\nP8.1.9 R2=%s dependency=afterany:%s\n' "$r1" "$r2" "$r1"
