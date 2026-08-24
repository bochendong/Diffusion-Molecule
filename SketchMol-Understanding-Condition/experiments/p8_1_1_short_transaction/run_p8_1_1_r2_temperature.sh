#!/usr/bin/env bash
set -euo pipefail

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
SEED="${P811_SEED:-7}"
R1_ROOT="${P811_R1_ROOT:-$PROJECT_DIR/outputs/p8_1_1_short_transaction_r1/seed_${SEED}}"
R2_ROOT="${P811_R2_ROOT:-$PROJECT_DIR/outputs/p8_1_1_short_transaction_r2_temperature/seed_${SEED}}"
P1_CHECKPOINT="${P811_P1_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
P6_ROOT="${P811_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
JOINT_ROOT="${P811_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
CHECKPOINT="$R1_ROOT/policy/umtp_graph_action_policy.pt"
TRAIN_CSV="$R1_ROOT/data/edit_train.csv"
EDIT_CSV="$P6_ROOT/data/edit_table1_gate.csv"
DENOVO_CSV="$P6_ROOT/data/denovo_hard_gate.csv"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"

for path in "$CHECKPOINT" "$TRAIN_CSV" "$EDIT_CSV" "$DENOVO_CSV" \
  "$R1_ROOT/eval/edit/sampling_summary.json" "$R1_ROOT/eval/denovo/sampling_summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing completed R1 artifact: $path" >&2; exit 2; }
done

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$SCRIPT_DIR/require_high_entropy.py" \
  --summary "$R1_ROOT/eval/edit/sampling_summary.json" --minimum "${P811_R2_MIN_ENTROPY:-0.85}"

mkdir -p "$R2_ROOT/eval/edit" "$R2_ROOT/eval/denovo"
# The de-novo arm is immutable because R2 changes only edit-policy temperature.
cp -a "$R1_ROOT/eval/denovo/." "$R2_ROOT/eval/denovo/"

echo "=== P8.1.1-R2 single factor: transaction temperature 1.0 -> 0.25 ==="
"$PYTHON_BIN" "$SCRIPT_DIR/sample_raw_transactions.py" \
  --checkpoint "$CHECKPOINT" --eval-csv "$EDIT_CSV" --eval-features-dir "$EDIT_FEATURES" \
  --output-csv "$R2_ROOT/eval/edit/candidates.csv" \
  --summary-json "$R2_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --temperature "${P811_R2_TEMPERATURE:-0.25}" --seed 2907 --device auto

for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EDIT_CSV" --candidates "$R2_ROOT/eval/edit/candidates.csv" \
    --output-dir "$R2_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_1_r2_temp_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EDIT_CSV" --candidates "$R2_ROOT/eval/edit/candidates.csv" \
  --output-dir "$R2_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p8_1_1_r2_temp_candidate20 --task-filter table1 --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/audit_unified_checkpoint.py" \
  --base-checkpoint "$P1_CHECKPOINT" --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_CSV" \
  --denovo-eval-csv "$DENOVO_CSV" --edit-eval-csv "$EDIT_CSV" \
  --denovo-summary "$R2_ROOT/eval/denovo/sampling_summary.json" \
  --edit-summary "$R2_ROOT/eval/edit/sampling_summary.json" \
  --output "$R2_ROOT/final_audit.json"
touch "$R2_ROOT/COMPLETE"
echo "P8.1.1-R2 complete: $R2_ROOT"

