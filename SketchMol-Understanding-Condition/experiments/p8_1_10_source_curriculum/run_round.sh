#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:?expected r1 or r2}"
[[ "$ROUND" == "r1" || "$ROUND" == "r2" ]] || { echo "round must be r1 or r2" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED=7
ROOT="$PROJECT_DIR/outputs/p8_1_10_source_curriculum/$ROUND/seed_${SEED}"
DATA="$ROOT/data"
TRAIN_ROOT="$PROJECT_DIR/outputs/direct_smiles_moledit_table1_group_rl_v1"
P6="$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}"
BASE="$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}/policy/unified_smiles_generator.pt"
ENTRY="$SCRIPT_DIR/source_curriculum_entrypoint.py"

export P817_SOURCE_CLAMP_SCALE=1.0
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$DATA" "$ROOT/stage1_reconstruction" "$ROOT/final_policy"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_curriculum.py" \
  --train "$TRAIN_ROOT/table1_train_pack/table1_benchmark_condition_rows.csv" \
  --eval "$P6/data/edit_table1_gate.csv" \
  --features "$TRAIN_ROOT/train_condition_features_hf_vlm" \
  --output "$DATA" --limit 2000 --seed "$SEED"

RECON="$DATA/reconstruction_clean.csv"
if [[ "$ROUND" == "r2" ]]; then
  RECON="$DATA/reconstruction_span_corrupt.csv"
fi

echo "Stage 1: source reconstruction; round=$ROUND input=$(basename "$RECON")"
"$PYTHON_BIN" "$ENTRY" train \
  --train-csv "$RECON" \
  --condition-features-dir "$TRAIN_ROOT/train_condition_features_hf_vlm" \
  --condition-layout unified \
  --resume-checkpoint "$BASE" --reset-training-state \
  --trainable-scope source_only --output-dir "$ROOT/stage1_reconstruction" \
  --epochs 1 --batch-size 64 --eval-batch-size 128 \
  --sampling-mode task_balanced --samples-per-epoch 2000 \
  --lr 1e-4 --weight-decay 0 --grad-clip 1.0 --seed "$SEED" --device auto

echo "Stage 2: identical train-only property-edit SFT for both rounds"
"$PYTHON_BIN" "$ENTRY" train \
  --train-csv "$DATA/property_edit.csv" \
  --condition-features-dir "$TRAIN_ROOT/train_condition_features_hf_vlm" \
  --condition-layout unified \
  --resume-checkpoint "$ROOT/stage1_reconstruction/unified_smiles_generator.pt" --reset-training-state \
  --trainable-scope source_only --output-dir "$ROOT/final_policy" \
  --epochs 1 --batch-size 64 --eval-batch-size 128 \
  --sampling-mode task_balanced --samples-per-epoch 2000 \
  --lr 5e-5 --weight-decay 0 --grad-clip 1.0 --seed "$SEED" --device auto

bash "$SCRIPT_DIR/run_eval.sh" "$ROUND" "$ROOT/final_policy/unified_smiles_generator.pt" "$ROOT"
