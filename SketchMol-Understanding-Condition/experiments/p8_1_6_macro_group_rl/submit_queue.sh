#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; LOG="$PROJECT_DIR/logs/p8_1_6_macro_group_rl"; mkdir -p "$LOG"
BASE_JOB="${P816_BASE_JOB:-20382124}"; COMMON=(--parsable --account="${P816_ACCOUNT:-rrg-hup}" --time="${P816_TIME:-00:20:00}" --mem=48G --cpus-per-task=8 --gpus="${P816_GPU:-h100:1}" --export=ALL)
R1="$(sbatch "${COMMON[@]}" --dependency="afterok:$BASE_JOB" --job-name=p8.1.6-r1-s7 --output="$LOG/r1-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r1 joint_bottleneck")"; R1="${R1%%;*}"
R2="$(sbatch "${COMMON[@]}" --dependency="afterany:$R1" --job-name=p8.1.6-r2-s7 --output="$LOG/r2-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r2 dense_softmin")"; R2="${R2%%;*}"
printf 'P8.1.6 submitted\nBASE=%s\nR1=%s aggregation=joint_bottleneck\nR2=%s aggregation=dense_softmin dependency=afterany:%s (same base restart)\n' "$BASE_JOB" "$R1" "$R2" "$R1"
