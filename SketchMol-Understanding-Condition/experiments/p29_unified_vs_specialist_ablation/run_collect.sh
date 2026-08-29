#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P29_SCRIPT_DIR:?P29_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="${P29_OUTPUT_ROOT:-$PROJECT/outputs/p29_unified_vs_specialist_ablation/seed_24003}"
python3 "$SCRIPT_DIR/collect_ablation.py" --root "$OUTPUT_ROOT"
touch "$OUTPUT_ROOT/ABLATION_COMPLETE"
