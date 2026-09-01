#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P37_SCRIPT_DIR:?P37_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P37_OUTPUT_ROOT:-$PROJECT/outputs/p37_denovo_overlap_expanded_eval/seed_37101}"
PY="${P37_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
EVAL_CSV="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv"
TRAIN_10K="$PROJECT/outputs/p33_joint_vs_separate_10k_single_seed/seed_33101/data/train.denovo.jsonl"
TRAIN_100K="$PROJECT/outputs/p35_joint_vs_separate_scale_sweep/scale_100000/seed_33101/data/train.denovo.jsonl"
GATE_10K="$PROJECT/outputs/p33_joint_vs_separate_10k_single_seed/seed_33101/data/gate.denovo.jsonl"
GATE_100K="$PROJECT/outputs/p35_joint_vs_separate_scale_sweep/scale_100000/seed_33101/data/gate.denovo.jsonl"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$SCRIPT_DIR:$PROJECT/experiments/p23_explicit_task_stage1_v2${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$SCRIPT_DIR/prepare_overlap_gate.py" --eval-csv "$EVAL_CSV" \
  --training-jsonl "$TRAIN_10K" --training-jsonl "$TRAIN_100K" \
  --prior-gate "$GATE_10K" --prior-gate "$GATE_100K" \
  --output-dir "$OUT/data" --seed 37101 --per-2p4p-cell 100 --per-5p-cell 40
