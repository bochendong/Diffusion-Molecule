#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P23_DENOVO_FILL_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/eval_denovo_table1_fill}"
OFFICIAL="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv"
P16="$PROJECT/outputs/p16_direct_llm_unified_generation_editing/seed_1616/data/train.mixed.jsonl"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717/data/train.paired.jsonl"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/data/train.sft.jsonl"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
for path in "$OFFICIAL" "$P16" "$P17" "$P23"; do
  [[ -f "$path" ]] || { echo "ERROR: missing de-novo fill input: $path" >&2; exit 2; }
done
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/data"
"$PY" "$SCRIPT_DIR/build_denovo_table1_fill.py" --eval-csv "$OFFICIAL" \
  --train-jsonl "$P16" --train-jsonl "$P17" --train-jsonl "$P23" \
  --output-dir "$OUT/data" --seed 2025 --conditions 100
touch "$OUT/DENOVO_FILL_PREPARED"
