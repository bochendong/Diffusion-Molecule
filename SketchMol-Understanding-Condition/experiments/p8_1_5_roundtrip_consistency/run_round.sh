#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:?r1 or r2}"
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
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P815_SEED:-7}"
OUT="${P815_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p8_1_5_roundtrip_${ROUND}/seed_${SEED}}"
DIRECT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
TABLE1="$PROJECT_DIR/outputs/direct_smiles_moledit_table1_group_rl_v1"
BASE="${P815_BASE_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
ENTRY="$SCRIPT_DIR/full_smiles_entrypoint.py"
mkdir -p "$OUT/data" "$OUT/policy"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_roundtrip_data.py" \
  --denovo-csv "$DIRECT/denovo_2p7p_train_rows.csv" \
  --edit-csv "$TABLE1/table1_train_pack/table1_benchmark_condition_rows.csv" \
  --denovo-features "$DIRECT/train_condition_features_hf_vlm" \
  --edit-features "$TABLE1/train_condition_features_hf_vlm" \
  --output-dir "$OUT/data" --per-mode "${P815_PER_MODE:-1000}" --seed "$SEED"

if [[ "$ROUND" == "r1" ]]; then
  TRAIN_CSV="$OUT/data/r1_forward.csv"
  CYCLE_WEIGHT=0
else
  TRAIN_CSV="$OUT/data/r2_forward_cycle.csv"
  CYCLE_WEIGHT=1
fi
echo "P8.1.5 $ROUND: cycle_weight=$CYCLE_WEIGHT; independent start from the same P1 checkpoint."
"$PYTHON_BIN" "$ENTRY" train --train-csv "$TRAIN_CSV" \
  --condition-features-dir "$OUT/data/features" --condition-layout unified \
  --resume-checkpoint "$BASE" --reset-training-state --allow-architecture-warmstart \
  --source-aware --source-encoder-layers 1 --source-adapter-layers 2 --source-adapter-bottleneck 64 \
  --trainable-scope source_only --output-dir "$OUT/policy" --epochs 2 --batch-size 64 --eval-batch-size 128 \
  --sampling-mode task_balanced --samples-per-epoch 4096 --lr 1e-4 --weight-decay 0 --grad-clip 1.0 \
  --seed "$SEED" --device auto
bash "$SCRIPT_DIR/run_common_eval.sh" "$ROUND" "$OUT/policy/unified_smiles_generator.pt" "$OUT"
