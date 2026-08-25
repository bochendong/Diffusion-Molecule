#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P17_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P17_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P17_OUTPUT_ROOT:-$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717}"
P16="$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616"
DENOVO_TRAIN="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_train_rows.csv"
EDIT_TRAIN="$PROJECT/outputs/p8_1_1_short_transaction_r1/seed_7/data/edit_train.csv"
P6="$PROJECT/outputs/p6_unified_transition_policy_v1/seed_7/data"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/data" "$OUT/pilot"
"$PY" "$SCRIPT_DIR/build_p17_data.py" \
  --denovo-csv "$DENOVO_TRAIN" --edit-csv "$EDIT_TRAIN" \
  --p16-train-jsonl "$P16/data/train.mixed.jsonl" --output-dir "$OUT/data" \
  --train-per-mode "${P17_TRAIN_PER_MODE:-160}" --dev-per-view-mode 32 --seed 1717
"$PY" "$SCRIPT_DIR/build_pilot_subsets.py" \
  --table1-csv "$P6/edit_table1_gate.csv" --denovo-csv "$P6/denovo_hard_gate.csv" \
  --p16-train-jsonl "$P16/data/train.mixed.jsonl" --p17-train-jsonl "$OUT/data/train.paired.jsonl" \
  --output-dir "$OUT/pilot" --seed 1717
touch "$OUT/PREPARED"
echo "P17 prepared: $OUT"
