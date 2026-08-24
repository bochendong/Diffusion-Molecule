#!/usr/bin/env bash
set -euo pipefail
ROUND="${1:?r1 or r2}"; [[ "$ROUND" == r1 || "$ROUND" == r2 ]] || exit 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7
P814="$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}"
CHECKPOINT="$P814/policy/unified_smiles_generator.pt"
OUT="$PROJECT_DIR/outputs/p8_1_7_source_clamped_policy/${ROUND}/seed_${SEED}"
P6="$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}"
DIRECT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
JOINT="$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2"
SCALE=1.0; [[ "$ROUND" == r2 ]] && SCALE=2.0
export P817_SOURCE_CLAMP_SCALE="$SCALE" PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/eval/denovo/raw" "$OUT/eval/edit/raw"
"$PYTHON_BIN" "$SCRIPT_DIR/preflight.py" --checkpoint "$CHECKPOINT" --p814-audit "$P814/final_audit.json" --output "$OUT/preflight.json"

sample() {
  local mode="$1" csv="$2" features="$3" seed="$4"
  "$PYTHON_BIN" "$SCRIPT_DIR/source_clamped_entrypoint.py" sample \
    --checkpoint "$CHECKPOINT" --eval-csv "$csv" --eval-condition-features-dir "$features" --condition-layout unified \
    --output-dir "$OUT/eval/$mode/raw" --prediction-csv "$OUT/eval/$mode/selected.csv" \
    --candidate-output-csv "$OUT/eval/$mode/candidates.csv" --method "p8_1_7_${ROUND}" \
    --decoding-mode sample --num-samples 20 --top-k-candidates 20 --max-candidates 20 --disable-finalizer \
    --smiles-grammar-constraint --max-new-tokens 120 --temperature 0.80 --top-k 32 --top-p 0.95 \
    --parallel-samples 10 --max-parallel-sequences 256 --seed "$seed" --device auto
}
sample denovo "$P6/data/denovo_hard_gate.csv" "$DIRECT/eval_condition_features_hf_vlm" 1817
sample edit "$P6/data/edit_table1_gate.csv" "$JOINT/feature_variants/validation_condition_features_hf_vlm" 2817

"$PYTHON_BIN" "$SCRIPT_DIR/normalize_denovo_candidates.py" --input "$OUT/eval/denovo/candidates.csv" --output "$OUT/eval/denovo/candidates_normalized.csv"
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$P6/data/denovo_hard_gate.csv" --candidates-csv "$OUT/eval/denovo/candidates_normalized.csv" \
  --output-json "$OUT/eval/denovo/metrics.json" --output-md "$OUT/eval/denovo/report.md" --budgets 1,8,20
for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" \
    --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_7_${ROUND}_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" \
  --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name "p8_1_7_${ROUND}_candidate20" --task-filter table1 --missing-oracle-policy fail
"$PYTHON_BIN" "$SCRIPT_DIR/audit_candidates.py" --checkpoint "$CHECKPOINT" --candidates "$OUT/eval/edit/candidates.csv" --scale "$SCALE" --output "$OUT/source_audit.json"
if [[ "$ROUND" == r2 ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/compare_rounds.py" --r1 "$PROJECT_DIR/outputs/p8_1_7_source_clamped_policy/r1/seed_${SEED}" --r2 "$OUT" --output "$OUT/comparison.json"
fi
touch "$OUT/COMPLETE"
