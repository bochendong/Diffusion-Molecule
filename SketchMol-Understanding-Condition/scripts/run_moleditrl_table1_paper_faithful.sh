#!/usr/bin/env bash
# Materialize the MolEditRL paper-faithful contract and optionally evaluate predictions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${MOLEDITRL_PYTHON_BIN:-${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}}"
CONTRACT_JSON="${MOLEDITRL_CONTRACT_JSON:-Research/Molecule Generation/OfficialBaselines/MolEditRL/paper_contract/moleditrl_table1_contract.json}"
OUTPUT_DIR="${MOLEDITRL_OUTPUT_DIR:-Research/Molecule Generation/OfficialBaselines/MolEditRL/results/table1_paper_faithful}"
PAPER_METRICS_CSV="${MOLEDITRL_PAPER_METRICS_CSV:-$OUTPUT_DIR/official_paper_metrics.csv}"
RUN_MANIFEST="${MOLEDITRL_RUN_MANIFEST:-$OUTPUT_DIR/official_run_manifest.json}"
REFERENCE_CSV="${MOLEDITRL_REFERENCE_CSV:-${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
PREDICTIONS_CSV="${MOLEDITRL_PREDICTIONS_CSV:-}"
PREDICTION_METHOD="${MOLEDITRL_PREDICTION_METHOD:-}"
EVAL_OUTPUT_DIR="${MOLEDITRL_EVAL_OUTPUT_DIR:-$OUTPUT_DIR/evaluated_predictions}"
MISSING_ORACLE_POLICY="${MOLEDITRL_MISSING_ORACLE_POLICY:-fail}"

if [[ ! -f "$CONTRACT_JSON" ]]; then
  echo "ERROR: MolEditRL paper contract not found: $CONTRACT_JSON" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "MolEditRL paper-faithful Table1 contract"
echo "  contract=$CONTRACT_JSON"
echo "  output_dir=$OUTPUT_DIR"
echo "  paper_metrics_csv=$PAPER_METRICS_CSV"
echo "  reference_csv=$REFERENCE_CSV"
echo "  predictions_csv=${PREDICTIONS_CSV:-none}"
echo "  missing_oracle_policy=$MISSING_ORACLE_POLICY"

"$PYTHON_BIN" - <<'PY' "$CONTRACT_JSON" "$PAPER_METRICS_CSV" "$RUN_MANIFEST" "$REFERENCE_CSV" "$PREDICTIONS_CSV" "$PREDICTION_METHOD"
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

contract_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
reference_csv = sys.argv[4]
predictions_csv = sys.argv[5]
prediction_method = sys.argv[6]

contract = json.loads(contract_path.read_text(encoding="utf-8"))
metrics_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
summary = contract.get("reported_summary", {})
for key, value in summary.items():
    rows.append({
        "row_type": "summary",
        "method": "MolEditRL",
        "source": "paper_contract",
        "task_key": "",
        "task": key,
        "acc_all_0_65": value if key.endswith("0_65") else "",
        "acc_all_0_15": value if key.endswith("0_15") else "",
        "status": "reported_by_paper",
    })
for task in contract.get("reported_table1_tasks", []):
    rows.append({
        "row_type": "task",
        "method": "MolEditRL",
        "source": "paper_contract",
        "task_key": task.get("task_key", ""),
        "task": task.get("label", ""),
        "acc_all_0_65": "" if task.get("mol_edit_rl_acc_all_0_65") is None else task.get("mol_edit_rl_acc_all_0_65"),
        "acc_all_0_15": "" if task.get("mol_edit_rl_acc_all_0_15") is None else task.get("mol_edit_rl_acc_all_0_15"),
        "status": "missing_task_value" if task.get("mol_edit_rl_acc_all_0_65") is None else "reported_by_paper",
    })

with metrics_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "row_type",
        "method",
        "source",
        "task_key",
        "task",
        "acc_all_0_65",
        "acc_all_0_15",
        "status",
    ])
    writer.writeheader()
    writer.writerows(rows)

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "benchmark_id": contract.get("benchmark_id"),
    "method_id": contract.get("method_id"),
    "reproduction_type": contract.get("reproduction_type"),
    "contract_json": str(contract_path),
    "paper_metrics_csv": str(metrics_path),
    "reference_csv": reference_csv,
    "predictions_csv": predictions_csv or None,
    "prediction_method": prediction_method or None,
    "status": "paper_faithful_reconstruction",
    "do_not_claim": contract.get("do_not_claim", []),
    "known_gaps": contract.get("known_gaps", []),
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"paper_metrics_csv": str(metrics_path), "manifest": str(manifest_path), "rows": len(rows)}, indent=2))
PY

if [[ -n "$PREDICTIONS_CSV" ]]; then
  if [[ ! -f "$REFERENCE_CSV" ]]; then
    echo "ERROR: reference CSV does not exist: $REFERENCE_CSV" >&2
    exit 2
  fi
  if [[ ! -f "$PREDICTIONS_CSV" ]]; then
    echo "ERROR: prediction CSV does not exist: $PREDICTIONS_CSV" >&2
    exit 2
  fi
  eval_args=(
    --reference "$REFERENCE_CSV"
    --predictions "$PREDICTIONS_CSV"
    --output-dir "$EVAL_OUTPUT_DIR"
    --model-name "MolEditRL-paper-faithful"
    --include-empty-table1
    --missing-oracle-policy "$MISSING_ORACLE_POLICY"
  )
  if [[ -n "$PREDICTION_METHOD" ]]; then
    eval_args+=(--method "$PREDICTION_METHOD")
  fi
  "$PYTHON_BIN" SketchMol-Unified-3MDiffusion/scripts/evaluate_moledit_table_metrics.py "${eval_args[@]}"
fi

echo
echo "MolEditRL paper-faithful contract ready:"
echo "  paper_metrics_csv=$PAPER_METRICS_CSV"
echo "  run_manifest=$RUN_MANIFEST"
if [[ -n "$PREDICTIONS_CSV" ]]; then
  echo "  evaluated_predictions=$EVAL_OUTPUT_DIR/moledit_table_summary.md"
fi
