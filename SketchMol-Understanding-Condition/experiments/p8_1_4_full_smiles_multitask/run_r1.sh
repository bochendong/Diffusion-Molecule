#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"; SEED="${P814_SEED:-7}"
OUT="${P814_R1_ROOT:-$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}}"
DIRECT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
TABLE1="$PROJECT_DIR/outputs/direct_smiles_moledit_table1_group_rl_v1"
BASE="${P814_BASE_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
UNIFIED="$PROJECT_DIR/experiments/unified_smiles_generator/unified_smiles_generator.py"
mkdir -p "$OUT/data" "$OUT/policy"; export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_mixed_data.py" \
  --denovo-csv "$DIRECT/denovo_2p7p_train_rows.csv" --edit-csv "$TABLE1/table1_train_pack/table1_benchmark_condition_rows.csv" \
  --denovo-features "$DIRECT/train_condition_features_hf_vlm" --edit-features "$TABLE1/train_condition_features_hf_vlm" \
  --output-dir "$OUT/data" --per-mode 2000 --r2-edit-limit 384 --seed "$SEED"

echo 'R1 factor: source-gated residual adapter SFT; no pointer/copy channel.'
"$PYTHON_BIN" "$UNIFIED" train --train-csv "$OUT/data/mixed_train.csv" \
  --condition-features-dir "$OUT/data/mixed_features" --condition-layout unified \
  --resume-checkpoint "$BASE" --reset-training-state --allow-architecture-warmstart \
  --source-aware --source-encoder-layers 1 --source-adapter-layers 2 --source-adapter-bottleneck 64 \
  --trainable-scope source_only --output-dir "$OUT/policy" --epochs 2 --batch-size 64 --eval-batch-size 128 \
  --sampling-mode task_balanced --samples-per-epoch 4096 --lr 1e-4 --weight-decay 0 --grad-clip 1.0 --seed "$SEED" --device auto
bash "$SCRIPT_DIR/run_common_eval.sh" r1 "$OUT/policy/unified_smiles_generator.pt" "$OUT"
