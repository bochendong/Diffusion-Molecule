#!/usr/bin/env bash
# Join Yansun pilot shard outputs (images + SMILES) into one deliverable CSV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

YANSUN_PILOT_OUTPUT_DIR="${YANSUN_PILOT_OUTPUT_DIR:-yansun/outputs/sketchmol_pilot_at5_v1}"
YANSUN_PILOT_EVAL_CSV="${YANSUN_PILOT_EVAL_CSV:-yansun/outputs/sketchmol_pilot_eval_rows.csv}"
YANSUN_PILOT_JOINED_CSV="${YANSUN_PILOT_JOINED_CSV:-$YANSUN_PILOT_OUTPUT_DIR/sketchmol_pilot_candidates.csv}"
YANSUN_PILOT_SUMMARY_JSON="${YANSUN_PILOT_SUMMARY_JSON:-$YANSUN_PILOT_OUTPUT_DIR/sketchmol_pilot_summary.json}"

SHARDS_DIR="$YANSUN_PILOT_OUTPUT_DIR/shards"
if [[ ! -d "$SHARDS_DIR" ]]; then
  echo "ERROR: shards dir not found: $SHARDS_DIR" >&2
  exit 2
fi

python3 "$SCRIPT_DIR/join_sketchmol_pilot_outputs.py" \
  --eval-csv "$YANSUN_PILOT_EVAL_CSV" \
  --shards-dir "$SHARDS_DIR" \
  --output-csv "$YANSUN_PILOT_JOINED_CSV" \
  --summary-json "$YANSUN_PILOT_SUMMARY_JSON"

echo "Joined pilot outputs:"
echo "  candidates_csv=$YANSUN_PILOT_JOINED_CSV"
echo "  summary_json=$YANSUN_PILOT_SUMMARY_JSON"
echo "  image_root=$YANSUN_PILOT_OUTPUT_DIR/shards"
