#!/usr/bin/env bash
# Yansun SketchMol pilot: one single-task + one two-property task, 5 candidates/row.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

YANSUN_SINGLE_TASK_ID="${YANSUN_SINGLE_TASK_ID:-101}"
YANSUN_MULTI_TASK_ID="${YANSUN_MULTI_TASK_ID:-201}"
YANSUN_PILOT_CANDIDATES="${YANSUN_PILOT_CANDIDATES:-5}"
YANSUN_PILOT_OUTPUT_DIR="${YANSUN_PILOT_OUTPUT_DIR:-yansun/outputs/sketchmol_pilot_at5_v1}"
YANSUN_PILOT_EVAL_CSV="${YANSUN_PILOT_EVAL_CSV:-yansun/outputs/sketchmol_pilot_eval_rows.csv}"
YANSUN_LOG_DIR="${YANSUN_LOG_DIR:-yansun/logs}"
mkdir -p "$YANSUN_LOG_DIR"

python3 "$SCRIPT_DIR/export_sketchmol_pilot_eval_rows.py" \
  --single-csv "$SCRIPT_DIR/sketchmol_targets_single.csv" \
  --multi-csv "$SCRIPT_DIR/sketchmol_targets_multi.csv" \
  --output-csv "$YANSUN_PILOT_EVAL_CSV" \
  --single-task-id "$YANSUN_SINGLE_TASK_ID" \
  --multi-task-id "$YANSUN_MULTI_TASK_ID"

export SKETCHMOL_DENOVO_EVAL_CSV="$YANSUN_PILOT_EVAL_CSV"
export SKETCHMOL_DENOVO_OUTPUT_DIR="$YANSUN_PILOT_OUTPUT_DIR"
export SKETCHMOL_CONDITIONAL_COUNT="$YANSUN_PILOT_CANDIDATES"
export SKETCHMOL_DENOVO_SHARD_COUNT="${YANSUN_PILOT_SHARD_COUNT:-10}"
export SKETCHMOL_DENOVO_SUBMIT_MODE="${YANSUN_PILOT_SUBMIT_MODE:-waves}"
export SKETCHMOL_DENOVO_WAVE_HOURS="${YANSUN_PILOT_WAVE_HOURS:-4}"
export SKETCHMOL_DENOVO_SUBMIT_EVAL=0
export SKETCHMOL_DENOVO_SLURM_JOB_NAME="yansun-sketchmol-pilot"
export SKETCHMOL_LOG_DIR="$YANSUN_LOG_DIR"

submit_output="$(
  bash SketchMolBenchmark/scripts/submit_denovo_2p7p_sketchmol_benchmark.sh 2>&1
)"
echo "$submit_output"

last_job_id="$(echo "$submit_output" | rg -o 'Submitted wave [0-9]+/[0-9]+: job=[0-9]+' | tail -1 | rg -o '[0-9]+$' || true)"
if [[ -z "$last_job_id" ]]; then
  last_job_id="$(echo "$submit_output" | rg -o 'Submitted (serial )?job: [0-9]+' | tail -1 | rg -o '[0-9]+$' || true)"
fi
if [[ -z "$last_job_id" ]]; then
  last_job_id="$(echo "$submit_output" | rg -o 'Submitted GPU job id\(s\):.*' | tail -1 | rg -o '[0-9]+' | tail -1 || true)"
fi
if [[ -z "$last_job_id" ]]; then
  echo "ERROR: failed to parse pilot GPU job id from submit output" >&2
  exit 1
fi

join_job_id="$(
  sbatch --parsable \
    --account="${SKETCHMOL_SLURM_ACCOUNT:-def-hup-ab}" \
    --job-name="yansun-sketchmol-pilot-join" \
    --time="00:30:00" \
    --mem="8G" \
    --cpus-per-task="2" \
    --dependency="afterok:${last_job_id}" \
    --output="$YANSUN_LOG_DIR/yansun-sketchmol-pilot-join-%j.log" \
    --export=ALL \
    --wrap="bash 'yansun/run_sketchmol_pilot_join.sh'"
)"

echo "Submitted join job: $join_job_id (depends on $last_job_id)"
echo "Pilot deliverables:"
echo "  images: $YANSUN_PILOT_OUTPUT_DIR/shards/"
echo "  joined_csv: $YANSUN_PILOT_OUTPUT_DIR/sketchmol_pilot_candidates.csv"
