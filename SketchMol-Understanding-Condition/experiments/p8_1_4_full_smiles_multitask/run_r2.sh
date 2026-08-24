#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"; SEED="${P814_SEED:-7}"
R1="$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}"; OUT="$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r2_grpo/seed_${SEED}"
UNIFIED="$PROJECT_DIR/experiments/unified_smiles_generator/unified_smiles_generator.py"; mkdir -p "$OUT/policy"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
echo 'R2 single factor: add one source-only GRPO epoch against identity copy.'
"$PYTHON_BIN" "$UNIFIED" group-rl --train-csv "$R1/data/r2_edit_train.csv" --condition-features-dir "$R1/data/mixed_features" \
  --condition-layout unified --resume-checkpoint "$R1/policy/unified_smiles_generator.pt" --trainable-scope source_only \
  --output-dir "$OUT/policy" --epochs 1 --limit 384 --batch-size 4 --lr 2e-6 --weight-decay 0 --rollouts-per-prompt 8 \
  --rl-objective grpo --grpo-update-epochs 1 --sft-weight 1.0 --reference-kl-weight 0.05 \
  --reward-mode table1_edit --reward-aggregation dense_softmin --reward-valid-weight 0.5 --reward-strict-weight 2.0 \
  --reward-source-similarity-weight 2.0 --reward-source-similarity-threshold 0.65 --reward-source-copy-penalty 1.0 \
  --smiles-grammar-constraint --max-new-tokens 100 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --parallel-samples 4 --max-parallel-sequences 256 --seed "$SEED" --device auto
bash "$SCRIPT_DIR/run_common_eval.sh" r2 "$OUT/policy/unified_smiles_generator_group_rl.pt" "$OUT"
