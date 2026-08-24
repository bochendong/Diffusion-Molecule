#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";PROJECT="$(cd "$HERE/../../.."&&pwd)";LOG="$PROJECT/logs/p8_2_state_adaptive_molprogram/transaction_group_rl";mkdir -p "$LOG";export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
common=(--parsable --account="${P82_ACCOUNT:-rrg-hup}" --time="${P82_TIME:-00:15:00}" --mem=48G --cpus-per-task=8 --gpus="${P82_GPU:-h100:1}" --export=ALL)
r1="$(sbatch "${common[@]}" --job-name=p82-trx-r1-s7 --output="$LOG/r1-%j.log" --wrap="bash '$HERE/run_round.sh' r1")";r1="${r1%%;*}"
r2="$(sbatch "${common[@]}" --dependency="afterany:$r1" --job-name=p82-trx-r2-s7 --output="$LOG/r2-%j.log" --wrap="bash '$HERE/run_round.sh' r2")";r2="${r2%%;*}"
printf 'P8.2 transaction R1=%s\nP8.2 transaction R2=%s dependency=afterany:%s\n' "$r1" "$r2" "$r1"
