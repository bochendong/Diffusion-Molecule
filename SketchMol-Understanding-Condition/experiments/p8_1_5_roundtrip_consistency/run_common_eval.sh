#!/usr/bin/env bash
set -euo pipefail

ROUND_NAME="${1:?round name}" CHECKPOINT="${2:?checkpoint}" OUTPUT_ROOT="${3:?output root}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENTRY="$SCRIPT_DIR/full_smiles_entrypoint.py"
SEED="${P815_SEED:-7}"
P6_ROOT="${P815_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
DIRECT_ROOT="${P815_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
JOINT_ROOT="${P815_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
BASE="${P815_BASE_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

mkdir -p "$OUTPUT_ROOT/eval/denovo" "$OUTPUT_ROOT/eval/edit"
sample_one() {
  local mode="$1" eval_csv="$2" features="$3" eval_seed="$4"
  "$PYTHON_BIN" "$ENTRY" sample \
    --checkpoint "$CHECKPOINT" --eval-csv "$eval_csv" --eval-condition-features-dir "$features" \
    --condition-layout unified --output-dir "$OUTPUT_ROOT/eval/$mode/raw" \
    --prediction-csv "$OUTPUT_ROOT/eval/$mode/selected_raw.csv" \
    --candidate-output-csv "$OUTPUT_ROOT/eval/$mode/candidates.csv" \
    --method "p8_1_5_${ROUND_NAME}" --decoding-mode sample --num-samples 20 \
    --top-k-candidates 20 --max-candidates 20 --disable-finalizer --smiles-grammar-constraint \
    --max-new-tokens 120 --temperature 0.80 --top-k 32 --top-p 0.95 \
    --parallel-samples 10 --max-parallel-sequences 256 --seed "$eval_seed" --device auto
}
sample_one denovo "$P6_ROOT/data/denovo_hard_gate.csv" "$DIRECT_ROOT/eval_condition_features_hf_vlm" 1815
sample_one edit "$P6_ROOT/data/edit_table1_gate.csv" "$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm" 2815

"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$P6_ROOT/data/denovo_hard_gate.csv" --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" --output-md "$OUTPUT_ROOT/eval/denovo/report.md" --budgets 1,8,20
for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$P6_ROOT/data/edit_table1_gate.csv" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_5_${ROUND_NAME}_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$P6_ROOT/data/edit_table1_gate.csv" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name "p8_1_5_${ROUND_NAME}_candidate20" --task-filter table1 --missing-oracle-policy fail
"$PYTHON_BIN" "$SCRIPT_DIR/audit_results.py" --base-checkpoint "$BASE" --checkpoint "$CHECKPOINT" \
  --edit-candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" --output "$OUTPUT_ROOT/final_audit.json"
touch "$OUTPUT_ROOT/COMPLETE"
