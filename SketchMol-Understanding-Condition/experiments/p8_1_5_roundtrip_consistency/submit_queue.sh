#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/p8_1_5_roundtrip_consistency"
mkdir -p "$LOG_DIR"
common=(--parsable --account="${P815_ACCOUNT:-rrg-hup}" --time="${P815_TIME:-00:15:00}" --mem=48G --cpus-per-task=8 --gpus="${P815_GPU:-h100:1}" --export=ALL)
r1="$(sbatch "${common[@]}" --job-name=p8.1.5-r1-s7 --output="$LOG_DIR/r1-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r1")"; r1="${r1%%;*}"
# R2 runs even if R1's diagnostic fails; it is an independently initialized causal arm.
r2="$(sbatch "${common[@]}" --dependency="afterany:$r1" --job-name=p8.1.5-r2-s7 --output="$LOG_DIR/r2-%j.log" --wrap="bash '$SCRIPT_DIR/run_round.sh' r2")"; r2="${r2%%;*}"
printf 'P8.1.5 queue submitted\nR1=%s\nR2=%s dependency=afterany:%s\n' "$r1" "$r2" "$r1"
