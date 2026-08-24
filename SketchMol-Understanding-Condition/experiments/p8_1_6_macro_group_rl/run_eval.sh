#!/usr/bin/env bash
set -euo pipefail
ROUND="${1:?round}" AGG="${2:?aggregation}" CHECKPOINT="${3:?checkpoint}" OUT="${4:?output}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"; SEED="${P816_SEED:-7}"
P6="$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}"; DIRECT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"; JOINT="$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2"; BASE="$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}/policy/unified_smiles_generator.pt"
mkdir -p "$OUT/eval/denovo" "$OUT/eval/edit"
sample() { local mode="$1" csv="$2" feat="$3" seed="$4"; "$PYTHON_BIN" "$SCRIPT_DIR/full_smiles_entrypoint.py" sample --checkpoint "$CHECKPOINT" --eval-csv "$csv" --eval-condition-features-dir "$feat" --condition-layout unified --output-dir "$OUT/eval/$mode/raw" --prediction-csv "$OUT/eval/$mode/selected.csv" --candidate-output-csv "$OUT/eval/$mode/candidates.csv" --method "p8_1_6_${ROUND}" --decoding-mode sample --num-samples 20 --top-k-candidates 20 --max-candidates 20 --disable-finalizer --smiles-grammar-constraint --max-new-tokens 120 --temperature 0.8 --top-k 32 --top-p 0.95 --parallel-samples 10 --max-parallel-sequences 256 --seed "$seed" --device auto; }
sample denovo "$P6/data/denovo_hard_gate.csv" "$DIRECT/eval_condition_features_hf_vlm" 1816
sample edit "$P6/data/edit_table1_gate.csv" "$JOINT/feature_variants/validation_condition_features_hf_vlm" 2816
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" --eval-csv "$P6/data/denovo_hard_gate.csv" --candidates-csv "$OUT/eval/denovo/candidates.csv" --output-json "$OUT/eval/denovo/metrics.json" --output-md "$OUT/eval/denovo/report.md" --budgets 1,8,20
for k in 1 8 20; do "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/any$k" --candidate-limit "$k" --model-name "p8_1_6_${ROUND}_any$k" --task-filter table1 --missing-oracle-policy fail; done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate --model-name "p8_1_6_${ROUND}_candidate20" --task-filter table1 --missing-oracle-policy fail
"$PYTHON_BIN" "$SCRIPT_DIR/audit_round.py" --base "$BASE" --checkpoint "$CHECKPOINT" --candidates "$OUT/eval/edit/candidates.csv" --expected-aggregation "$AGG" --output "$OUT/final_audit.json"
touch "$OUT/COMPLETE"
