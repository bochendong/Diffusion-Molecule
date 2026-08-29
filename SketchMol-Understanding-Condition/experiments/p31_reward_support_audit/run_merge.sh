#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P31_SCRIPT_DIR:?P31_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P31_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P31_OUTPUT_ROOT:-$PROJECT/outputs/p31_reward_support_audit/seed_31001}"
inputs=()
for shard in 0 1 2 3; do
  marker=$(printf '%s/SHARD_%02d_COMPLETE' "$OUT" "$shard")
  path=$(printf '%s/shards/shard-%02d.candidates.jsonl' "$OUT" "$shard")
  test -f "$marker"; test -f "$path"
  inputs+=("$path")
done
"$PY" "$SCRIPT_DIR/merge_audit.py" --inputs "${inputs[@]}" --output-dir "$OUT/audit"
touch "$OUT/AUDIT_COMPLETE"
