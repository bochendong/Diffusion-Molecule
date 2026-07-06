#!/usr/bin/env bash
# Merge official GeLLMO-C task CSVs, build oracle properties, and evaluate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-$PYTHON_BIN}"
OUTPUT_DIR="${SUCC_GELLMOC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_gellmoc_official_cmumo_v1}"
TASKS="${SUCC_GELLMOC_TASKS:-BPQ,ELQ,ACEP,BDPQ,DHMQ,CDE,ABMP,BCMQ,BDEQ,HLMPQ}"
SETTINGS="${SUCC_GELLMOC_SETTINGS:-seen}"
MERGED_CSV="${SUCC_GELLMOC_MERGED_PREDICTION_CSV:-$OUTPUT_DIR/gellmoc_candidate_predictions.csv}"
ORACLE_OUTPUT="${SUCC_GELLMOC_ORACLE_OUTPUT_CSV:-$OUTPUT_DIR/generated_properties.csv}"
ORACLE_WORK_DIR="${SUCC_GELLMOC_ORACLE_WORK_DIR:-$OUTPUT_DIR/oracle_work}"
DEFAULT_MERGE_PROPERTIES_CSV="SketchMol-Understanding-Condition/outputs/external_oracle_build_cmumo_v1/generated_properties.csv"
MERGE_PROPERTIES_CSV="${SUCC_GELLMOC_MERGE_PROPERTIES_CSV:-}"
if [[ -z "$MERGE_PROPERTIES_CSV" && -f "$DEFAULT_MERGE_PROPERTIES_CSV" ]]; then
  MERGE_PROPERTIES_CSV="$DEFAULT_MERGE_PROPERTIES_CSV"
fi
EVAL_OUTPUT_DIR="${SUCC_GELLMOC_EVAL_OUTPUT_DIR:-$OUTPUT_DIR/eval_official_oracle}"
SKIP_ORACLE="${SUCC_GELLMOC_SKIP_ORACLE:-0}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

echo "Official GeLLMO-C postprocess"
echo "  output_dir=$OUTPUT_DIR"
echo "  tasks=$TASKS"
echo "  settings=$SETTINGS"
echo "  merged_csv=$MERGED_CSV"
echo "  oracle_output=$ORACLE_OUTPUT"
echo "  merge_properties_csv=${MERGE_PROPERTIES_CSV:-none}"

IFS=',' read -r -a task_array <<< "$TASKS"
IFS=',' read -r -a setting_array <<< "$SETTINGS"
candidate_csvs=()
for task in "${task_array[@]}"; do
  task_trimmed="${task#"${task%%[![:space:]]*}"}"
  task_trimmed="${task_trimmed%"${task_trimmed##*[![:space:]]}"}"
  [[ -z "$task_trimmed" ]] && continue
  task_slug="${task_trimmed//+/_}"
  for setting in "${setting_array[@]}"; do
    setting_trimmed="${setting#"${setting%%[![:space:]]*}"}"
    setting_trimmed="${setting_trimmed%"${setting_trimmed##*[![:space:]]}"}"
    [[ -z "$setting_trimmed" ]] && continue
    csv_path="$OUTPUT_DIR/tasks/${task_slug}_${setting_trimmed}/gellmoc_candidate_predictions.csv"
    if [[ ! -f "$csv_path" ]]; then
      echo "ERROR: missing task candidate CSV: $csv_path" >&2
      exit 2
    fi
    candidate_csvs+=("$csv_path")
  done
done

"$PYTHON_BIN" - <<'PY' "$MERGED_CSV" "${candidate_csvs[@]}"
from __future__ import annotations

import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
paths = [Path(item) for item in sys.argv[2:]]
rows = []
fieldnames = []
seen = set()
for path in paths:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for field in reader.fieldnames or []:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
        rows.extend(dict(row) for row in reader)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(paths)} files -> {out} rows={len(rows)}")
PY

if [[ "$SKIP_ORACLE" != "1" ]]; then
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$MERGED_CSV" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_OUTPUT" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_WORK_DIR" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$MERGE_PROPERTIES_CSV" \
  bash "$PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
  --prediction-csv "$MERGED_CSV" \
  --output-dir "$EVAL_OUTPUT_DIR" \
  --generated-properties-csv "$ORACLE_OUTPUT" \
  --source-properties-csv "$ORACLE_OUTPUT" \
  --smiles-column generated_smiles \
  --source-smiles-column source_smiles \
  --group-column condition_id \
  --report-title "Official GeLLMO-C C-MuMO Baseline"

RUN_MANIFEST="$OUTPUT_DIR/official_run_manifest.json"
"$PYTHON_BIN" - <<'PY' "$RUN_MANIFEST" "$MERGED_CSV" "$ORACLE_OUTPUT" "$EVAL_OUTPUT_DIR" "$TASKS" "$SETTINGS"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "benchmark_id": "cmumo",
    "method_id": "GeLLMO-C",
    "reproduction_type": "official_code",
    "tasks": sys.argv[5],
    "settings": sys.argv[6],
    "merged_prediction_csv": sys.argv[2],
    "oracle_csv": sys.argv[3],
    "eval_output_dir": sys.argv[4],
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo
echo "Official GeLLMO-C postprocess ready:"
echo "  merged_csv=$MERGED_CSV"
echo "  oracle_csv=$ORACLE_OUTPUT"
echo "  report=$EVAL_OUTPUT_DIR/external_multiproperty_report.md"
echo "  run_manifest=$RUN_MANIFEST"
