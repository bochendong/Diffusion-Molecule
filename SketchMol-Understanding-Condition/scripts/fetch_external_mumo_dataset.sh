#!/usr/bin/env bash
# Download NingLab/MuMOInstruct JSON splits for SUCC external benchmarks.
#
# Output layout:
#   $DM_DATA_ROOT/external/mumo/train.json
#   $DM_DATA_ROOT/external/mumo/test.json
#
# Usage:
#   bash SketchMol-Understanding-Condition/scripts/fetch_external_mumo_dataset.sh

set -euo pipefail

DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
OUTPUT_DIR="${SUCC_FETCH_MUMO_OUTPUT_DIR:-$DM_DATA_ROOT/external/mumo}"
HF_DATASET="NingLab/MuMOInstruct"
DRY_RUN="${SUCC_FETCH_MUMO_DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR"

echo "MuMOInstruct fetch"
echo "  dataset=$HF_DATASET"
echo "  output_dir=$OUTPUT_DIR"
echo "  dry_run=$DRY_RUN"

fetch_split() {
  local split="$1"
  local gz_name="${split}.json.gz"
  local url="https://huggingface.co/datasets/${HF_DATASET}/resolve/main/${gz_name}"
  local gz_path="$OUTPUT_DIR/${gz_name}"
  local json_path="$OUTPUT_DIR/${split}.json"

  if [[ -s "$json_path" ]]; then
    echo "exists: $json_path"
    return
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "would download: $url -> $json_path"
    return
  fi

  if [[ ! -s "$gz_path" ]]; then
    echo "download: $url"
    curl -L --fail --continue-at - --output "$gz_path" "$url"
  fi

  echo "decompress: $gz_path -> $json_path"
  gunzip -c "$gz_path" > "$json_path"
}

fetch_split train
fetch_split test

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry-run complete"
  exit 0
fi

python3 - <<'PY' "$OUTPUT_DIR/test.json" "$OUTPUT_DIR/train.json"
import json
import sys
from pathlib import Path

for path_str in sys.argv[1:]:
    path = Path(path_str)
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path.name}: rows={len(data)} type={type(data).__name__}")
PY

echo
echo "Done."
echo "  train: $OUTPUT_DIR/train.json"
echo "  test: $OUTPUT_DIR/test.json"
