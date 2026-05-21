#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE_ROOT="${MOLPILOT_STAGE_ROOT:-${MOLPILOT_RUN_DIR:-}}"
if [[ -z "$STAGE_ROOT" ]]; then
  echo "ERROR: set MOLPILOT_STAGE_ROOT to an existing outputs/stages/<run> directory."
  exit 2
fi

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

SUFFIXES="${MOLPILOT_ABLATION_SUFFIXES:-ablate_full_graph ablate_graph_only ablate_scaffold_library_only ablate_latent_only ablate_diffusion_only ablate_full_no_library ablate_full_wide ablate_graph_heavy ablate_scaffold_library_heavy ablate_full_heavy ablate_full_heavy_seed17}"
OUT="${MOLPILOT_ABLATION_SUMMARY:-$STAGE_ROOT/resample_ablation_summary.csv}"

"$PYTHON_BIN" - "$STAGE_ROOT" "$OUT" $SUFFIXES <<'PY'
import csv
import json
import sys
from pathlib import Path

stage_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
suffixes = sys.argv[3:]
fields = [
    "suffix",
    "status",
    "requests",
    "candidates",
    "request_overall_at_10",
    "macro_task_request_overall_at_10",
    "task_edit_request_overall_at_10",
    "task_inpaint_request_overall_at_10",
    "task_de_novo_request_overall_at_10",
    "origin_diffusion_overall_success",
    "origin_source_guided_0_25_overall_success",
    "origin_source_neighborhood_overall_success",
    "origin_scaffold_library_overall_success",
    "origin_family_diffusion_overall_success",
    "origin_family_source_guided_overall_success",
    "origin_family_source_neighborhood_overall_success",
    "origin_family_graph_edit_overall_success",
    "origin_family_graph_grow_overall_success",
    "origin_family_scaffold_library_overall_success",
    "failure_reason_scaffold_changed",
    "failure_reason_mw_drift",
    "failure_reason_low_similarity",
]
rows = []
for suffix in suffixes:
    metrics_path = stage_root / f"eval_metrics_{suffix}.json"
    row = {"suffix": suffix, "status": "missing"}
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        row["status"] = "ok"
        for field in fields:
            if field in {"suffix", "status"}:
                continue
            row[field] = metrics.get(field, "")
    rows.append(row)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_path}")
for row in rows:
    print(
        row.get("suffix", ""),
        row.get("status", ""),
        "edit@10=", row.get("task_edit_request_overall_at_10", ""),
        "inpaint@10=", row.get("task_inpaint_request_overall_at_10", ""),
        "macro@10=", row.get("macro_task_request_overall_at_10", ""),
        "graph_family=", row.get("origin_family_graph_edit_overall_success", ""),
        "library_family=", row.get("origin_family_scaffold_library_overall_success", ""),
    )
PY
