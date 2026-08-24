#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/p8_1_3_macro_graph"
mkdir -p "$LOG_DIR"
R1="$(sbatch --parsable --job-name=p8.1.3-r1-macro --time=00:10:00 --cpus-per-task=4 --mem=8G --account=rrg-bengioy-ad --output="$LOG_DIR/r1-%j.out" --error="$LOG_DIR/r1-%j.err" --chdir="$REPO_DIR" --wrap="bash '$SCRIPT_DIR/run_p8_1_3_r1.sh'")"
R2="$(sbatch --parsable --dependency=afterok:${R1} --job-name=p8.1.3-r2-brics --time=00:10:00 --cpus-per-task=4 --mem=8G --account=rrg-bengioy-ad --output="$LOG_DIR/r2-%j.out" --error="$LOG_DIR/r2-%j.err" --chdir="$REPO_DIR" --wrap="bash '$SCRIPT_DIR/run_p8_1_3_r2.sh'")"
printf 'p8_1_3_r1_job=%s\np8_1_3_r2_job=%s\n' "$R1" "$R2"
