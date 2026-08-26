#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
OUT="${P23_EDIT500_OUTPUT_ROOT:-$TRAIN_OUT/eval_moledit_table1_500}"
HELDOUT="${P23_EDIT500_HELDOUT_CSV:-/scratch/bdong/datasets/Diffusion-Molecule/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/data"
"$PY" "$SCRIPT_DIR/build_edit_table1_500.py" --heldout-csv "$HELDOUT" \
  --train-jsonl "$TRAIN_OUT/data/train.sft.jsonl" --output-dir "$OUT/data" \
  --rows-per-task 500 --candidate-pool-per-task 5000 --seed 23500
touch "$OUT/EDIT500_PREPARED"
