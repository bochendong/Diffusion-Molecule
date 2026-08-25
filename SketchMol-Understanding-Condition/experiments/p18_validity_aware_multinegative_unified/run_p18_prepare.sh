#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P18_SCRIPT_DIR:?P18_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"; PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P18_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P18_OUTPUT_ROOT:-$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818}"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; fi
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/data"
"$PY" "$SCRIPT_DIR/audit_locked_inputs.py" --p17-output "$P17" --preregistration "$SCRIPT_DIR/preregistration.json" --output "$OUT/data/locked_input_audit.json"
"$PY" "$SCRIPT_DIR/build_multinegatives.py" --p17-train "$P17/data/train.paired.jsonl" --output-jsonl "$OUT/data/train.multinegative.jsonl" --manifest "$OUT/data/manifest.json"
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
touch "$OUT/PREPARED"
echo "P18 prepared: $OUT"
