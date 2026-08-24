#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../../.." && pwd)"; LOG="$PROJECT/logs/p8_2_transaction_group_rl"; mkdir -p "$LOG"
R2_JOB_ID="${P82_R2_JOB_ID:-20396370}"
job="$(sbatch --parsable --account="${P82_ACCOUNT:-rrg-hup}" --time=00:08:00 --mem=48G --cpus-per-task=8 --gpus="${P82_GPU:-h100:1}" --export=ALL --dependency="afterok:$R2_JOB_ID" --job-name=p82-trx-base-replay-s7 --output="$LOG/base-replay-%j.log" --wrap="bash '$HERE/run_base_replay.sh'")"; job="${job%%;*}"
printf 'P8.2 transaction base replay=%s dependency=afterok:%s\n' "$job" "$R2_JOB_ID"
