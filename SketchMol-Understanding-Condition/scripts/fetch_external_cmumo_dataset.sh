#!/usr/bin/env bash
# Download and normalize NingLab/C-MuMOInstruct test splits for SUCC external benchmarks.
#
# Output layout (matches official suite defaults):
#   $DM_DATA_ROOT/external/cmumo/test.json
#   $DM_DATA_ROOT/external/cmumo/train.json   (optional)
#
# Usage:
#   bash SketchMol-Understanding-Condition/scripts/fetch_external_cmumo_dataset.sh
#   SUCC_FETCH_CMUMO_INCLUDE_TRAIN=1 bash .../fetch_external_cmumo_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
OUTPUT_DIR="${SUCC_FETCH_CMUMO_OUTPUT_DIR:-$DM_DATA_ROOT/external/cmumo}"
HF_DATASET="NingLab/C-MuMOInstruct"
INCLUDE_TRAIN="${SUCC_FETCH_CMUMO_INCLUDE_TRAIN:-0}"
DRY_RUN="${SUCC_FETCH_CMUMO_DRY_RUN:-0}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

TEST_FILES=(
  test_abmp_mixed.json.gz
  test_acep_mixed.json.gz
  test_bcmq_mixed.json.gz
  test_bdeq_mixed.json.gz
  test_bdpq_mixed.json.gz
  test_bpq_mixed.json.gz
  test_cde_mixed.json.gz
  test_dhmq_mixed.json.gz
  test_elq_mixed.json.gz
  test_hlmpq_mixed.json.gz
)

mkdir -p "$OUTPUT_DIR/raw"

echo "C-MuMOInstruct fetch"
echo "  dataset=$HF_DATASET"
echo "  output_dir=$OUTPUT_DIR"
echo "  include_train=$INCLUDE_TRAIN"
echo "  dry_run=$DRY_RUN"

download_if_missing() {
  local filename="$1"
  local url="https://huggingface.co/datasets/${HF_DATASET}/resolve/main/${filename}"
  local output="$OUTPUT_DIR/raw/${filename}"
  if [[ -s "$output" ]]; then
    echo "exists: $output"
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "would download: $url -> $output"
    return
  fi
  echo "download: $url"
  curl -L --fail --continue-at - --output "$output" "$url"
}

for filename in "${TEST_FILES[@]}"; do
  download_if_missing "$filename"
done

if [[ "$INCLUDE_TRAIN" == "1" ]]; then
  download_if_missing "train.json.gz"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry-run complete"
  exit 0
fi

"$PYTHON_BIN" - <<'PY' "$OUTPUT_DIR"
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

output_dir = Path(sys.argv[1])
raw_dir = output_dir / "raw"
test_files = sorted(raw_dir.glob("test_*_mixed.json.gz"))

rows: list[dict] = []
for path in test_files:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}, got {type(payload).__name__}")
    for item in payload:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        property_comb = str(row.get("property_comb") or "").strip().lower()
        task = str(row.get("task") or "").strip()
        if property_comb:
            row["task"] = property_comb
        elif task.lower().startswith("c:"):
            row["task"] = task.split(":", 1)[1].strip().lower()
        rows.append(row)

if not rows:
    raise SystemExit(f"No C-MuMO rows found under {raw_dir}")

test_path = output_dir / "test.json"
test_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")

train_gz = raw_dir / "train.json.gz"
if train_gz.exists():
    with gzip.open(train_gz, "rt", encoding="utf-8") as handle:
        train_payload = json.load(handle)
    if isinstance(train_payload, list):
        train_rows = []
        for item in train_payload:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            property_comb = str(row.get("property_comb") or "").strip().lower()
            task = str(row.get("task") or "").strip()
            if property_comb:
                row["task"] = property_comb
            elif task.lower().startswith("c:"):
                row["task"] = task.split(":", 1)[1].strip().lower()
            train_rows.append(row)
        (output_dir / "train.json").write_text(json.dumps(train_rows, ensure_ascii=False) + "\n", encoding="utf-8")

counts = Counter((row.get("task"), row.get("split"), row.get("instr_setting")) for row in rows)
print(f"wrote {test_path} rows={len(rows)} unique_tasks={len({row.get('task') for row in rows})}")
for key, value in sorted(counts.items()):
    print(f"  task={key[0]} split={key[1]} setting={key[2]} rows={value}")
PY

echo
echo "Done."
echo "  test: $OUTPUT_DIR/test.json"
if [[ -f "$OUTPUT_DIR/train.json" ]]; then
  echo "  train: $OUTPUT_DIR/train.json"
fi
echo
echo "Next: run C-MuMO official suite with"
echo "  SUCC_OFFICIAL_RUN_MUMO=0 \\"
echo "  SUCC_OFFICIAL_CMUMO_SOURCE_FILE=$OUTPUT_DIR/test.json \\"
echo "  bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh"
