#!/usr/bin/env bash
# Print a compact side-by-side table for a torch sweep.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_NAME="${SKETCHIMAGE_SWEEP_NAME:-${1:-sketchmol_aligned_torch_50k_10k_v10_contrastive_temp}}"
VARIANTS="${SKETCHIMAGE_SWEEP_VARIANTS:-contrastive_cool contrastive_cooler contrastive_cold}"
PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

"$PYTHON_BIN" - "$SWEEP_NAME" "$VARIANTS" <<'PY'
import csv
import sys
from pathlib import Path

sweep_name = sys.argv[1]
variants = sys.argv[2].split()
fields = [
    "run",
    "task_type",
    "n",
    "top1_target_tanimoto",
    "mean_best_tanimoto",
    "topk_target_hit",
    "top1_scaffold_match",
    "top1_property_success",
    "topk_property_success",
]

rows = []
missing = []
for variant in variants:
    run_name = f"{sweep_name}_{variant}"
    summary_path = Path("outputs/runs") / run_name / "task_type_summary.csv"
    if not summary_path.exists():
        missing.append(str(summary_path))
        continue
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("task_type") in {"overall", "de_novo", "edit", "fragment_grow", "inpaint"}:
                out = {field: row.get(field, "") for field in fields}
                out["run"] = variant
                rows.append(out)

writer = csv.DictWriter(sys.stdout, fieldnames=fields)
writer.writeheader()
for row in rows:
    writer.writerow(row)

if missing:
    print("", file=sys.stderr)
    print("Missing summaries:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
PY
