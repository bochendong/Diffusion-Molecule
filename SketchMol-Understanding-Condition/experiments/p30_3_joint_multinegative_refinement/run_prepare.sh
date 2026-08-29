#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P303_SCRIPT_DIR:?P303_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P303_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
OUT="${P303_OUTPUT_ROOT:-$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301}"
test -f "$P23/data/train.contrastive.jsonl"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$SCRIPT_DIR/build_balanced_invalid_refinement.py" \
  --input-jsonl "$P23/data/train.contrastive.jsonl" \
  --output-jsonl "$OUT/data/train.invalid_balanced.jsonl" \
  --manifest "$OUT/data/manifest.json" \
  --denovo-per-arity 100 --edit-per-task 60 --seed 30301
touch "$OUT/PREPARED"
