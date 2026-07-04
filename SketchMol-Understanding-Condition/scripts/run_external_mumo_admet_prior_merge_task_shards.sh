#!/usr/bin/env bash
# Merge per-task MuMO GraphEditDSL shard outputs into the main benchmark CSVs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="${SUCC_MUMO_ADMET_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_graph_edit_admet_prior_v1}"
ROWS_CSV="${SUCC_MUMO_ADMET_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_rows.csv}"
TASK_LIST_FILE="${SUCC_MUMO_ADMET_TASK_LIST_FILE:-$OUTPUT_DIR/parallel_task_list.txt}"
if [[ -f "$TASK_LIST_FILE" ]]; then
  TASKS="$(tr '\n' ',' < "$TASK_LIST_FILE" | sed 's/,$//')"
elif [[ -n "${SUCC_MUMO_ADMET_TASK_LIST:-}" ]]; then
  TASKS="${SUCC_MUMO_ADMET_TASK_LIST//:/,}"
else
  TASKS="BDP,BDQ,BPQ,DPQ,BDPQ,MPQ,BDMQ,BHMQ,BMPQ,HMPQ"
fi
BENCHMARK_OUTPUT_DIR="${SUCC_MUMO_ADMET_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_graph_edit_agent}"
PREDICTION_CSV="${SUCC_MUMO_ADMET_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/graph_edit_agent_predictions.csv}"
CANDIDATE_PREDICTION_CSV="${SUCC_MUMO_ADMET_CANDIDATE_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/graph_edit_agent_candidate_predictions.csv}"
PLAN_JSONL="${SUCC_MUMO_ADMET_PLAN_JSONL:-$OUTPUT_DIR/graph_edit_plans.jsonl}"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
SHARD_ARGS=()
for task in "${TASK_ARRAY[@]}"; do
  task="$(echo "$task" | xargs)"
  [[ -n "$task" ]] || continue
  SHARD_ARGS+=(--shard-dir "$OUTPUT_DIR/tasks/$task")
done

if [[ ! -f "$ROWS_CSV" ]]; then
  echo "ERROR: missing rows CSV for merge: $ROWS_CSV" >&2
  exit 2
fi
if [[ ${#SHARD_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: no task shards configured for merge." >&2
  exit 2
fi

echo "Merging MuMO GraphEdit task shards"
echo "  python=$PYTHON_BIN"
echo "  rows_csv=$ROWS_CSV"
echo "  tasks=$TASKS"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  candidate_csv=$CANDIDATE_PREDICTION_CSV"
echo "  plan_jsonl=$PLAN_JSONL"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/merge_external_graph_edit_task_shards.py" \
  --rows-csv "$ROWS_CSV" \
  "${SHARD_ARGS[@]}" \
  --prediction-csv "$PREDICTION_CSV" \
  --candidate-output-csv "$CANDIDATE_PREDICTION_CSV" \
  --plan-jsonl "$PLAN_JSONL"

echo
echo "MuMO GraphEdit shard merge ready:"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  candidate_csv=$CANDIDATE_PREDICTION_CSV"
echo "  plan_jsonl=$PLAN_JSONL"
