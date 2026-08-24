#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; LOG="$PROJECT/logs/p8_2_matched_inference"; mkdir -p "$LOG"; export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
cpu=(--parsable --account="${P82_ACCOUNT:-def-hup-ab}" --mem=32G --cpus-per-task=8 --export=ALL)
pre="$(sbatch "${cpu[@]}" --time=00:05:00 --job-name=p82-pre-s7 --output="$LOG/pre-%j.log" --wrap="bash '$HERE/run_pre.sh'")"; pre="${pre%%;*}"
table="$(sbatch "${cpu[@]}" --time=00:10:00 --dependency="afterok:$pre" --job-name=p82-table-s7 --output="$LOG/table-%j.log" --wrap="bash '$HERE/run_table1.sh'")"; table="${table%%;*}"
denovo="$(sbatch --parsable --account="${P82_ACCOUNT:-def-hup-ab}" --time="${P82_DENOVO_TIME:-00:20:00}" --mem=48G --cpus-per-task=8 --gpus="${P82_GPU:-h100:1}" --export=ALL --array=2-7%6 --dependency="afterok:$pre" --job-name=p82-2p7p-s7 --output="$LOG/denovo-%A_%a.log" --wrap="bash '$HERE/run_denovo_shard.sh' \$SLURM_ARRAY_TASK_ID")"; denovo="${denovo%%;*}"
final="$(sbatch "${cpu[@]}" --time=00:10:00 --dependency="afterok:$table:$denovo" --job-name=p82-final-s7 --output="$LOG/final-%j.log" --wrap="bash '$HERE/finalize.sh'")"; final="${final%%;*}"
printf 'P8.2 PRE=%s\nP8.2 Table1=%s dependency=afterok:%s\nP8.2 2p7p-array=%s[2-7] dependency=afterok:%s\nP8.2 FINAL=%s dependency=afterok:%s:%s\n' "$pre" "$table" "$pre" "$denovo" "$pre" "$final" "$table" "$denovo"
