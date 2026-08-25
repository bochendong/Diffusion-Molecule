#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P21_SCRIPT_DIR:?P21_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P21_OUTPUT_ROOT:-$PROJECT/outputs/p21_unified_program_grpo/seed_2121}"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P18="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12"
P19="$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; fi
test -f "$P17/data/train.paired.jsonl"
test -f "$P18/model/p18/adapter/adapter_model.safetensors"
test -f "$P19/COMPLETE"
actual="$(sha256sum "$P18/model/p18/adapter/adapter_model.safetensors" | awk '{print $1}')"
test "$actual" = "7b3e1736bac49b7b2e35eceeed11fde199e0403a948e4dae08fd4fb9b89a0827"
mkdir -p "$OUT"
cp "$SCRIPT_DIR/preregistration.json" "$OUT/preregistration.snapshot.json"
python -m pytest -q "$SCRIPT_DIR/test_contract.py"
touch "$OUT/PREPARED"
echo "P21 locked inputs verified"
