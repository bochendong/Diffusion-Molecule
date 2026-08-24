#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/p8_1_4_full_smiles_multitask"; mkdir -p "$LOG_DIR"
common=(--parsable --account="${P814_ACCOUNT:-rrg-hup}" --time="${P814_TIME:-00:20:00}" --mem=48G --cpus-per-task=8 --gpus="${P814_GPU:-h100:1}" --export=ALL)
r1="$(sbatch "${common[@]}" --job-name=p8.1.4-r1-s7 --output="$LOG_DIR/r1-%j.log" --wrap="bash '$SCRIPT_DIR/run_r1.sh'")"; r1="${r1%%;*}"
r2="$(sbatch "${common[@]}" --dependency="afterok:$r1" --job-name=p8.1.4-r2-s7 --output="$LOG_DIR/r2-%j.log" --wrap="bash '$SCRIPT_DIR/run_r2.sh'")"; r2="${r2%%;*}"
printf 'P8.1.4 queue submitted\nR1=%s\nR2=%s dependency=afterok:%s\n' "$r1" "$r2" "$r1"
