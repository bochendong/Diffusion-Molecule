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
SEED="${P818_SEED:-7}"
R1_ROOT="${P818_R1_ROOT:-$PROJECT_DIR/outputs/p8_1_8_masked_molecule_policy_r1/seed_${SEED}}"
R2_ROOT="${P818_R2_ROOT:-$PROJECT_DIR/outputs/p8_1_8_masked_molecule_policy_r2/seed_${SEED}}"
P6_ROOT="${P818_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
JOINT_ROOT="${P818_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
CHECKPOINT="$R1_ROOT/policy/masked_molecule_policy.pt"
EDIT_CSV="$P6_ROOT/data/edit_table1_gate.csv"
DENOVO_CSV="$P6_ROOT/data/denovo_hard_gate.csv"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"
R2_FRACTION="${P818_R2_MASK_FRACTION:-$(
  "$PYTHON_BIN" "$SCRIPT_DIR/choose_r2_mask_fraction.py" \
    --summary "$R1_ROOT/eval/edit/sampling_summary.json"
)}"

mkdir -p "$R2_ROOT/eval/edit" "$R2_ROOT/eval/denovo"
cp -a "$R1_ROOT/eval/denovo/." "$R2_ROOT/eval/denovo/"
"$PYTHON_BIN" "$SCRIPT_DIR/normalize_denovo_candidates.py" \
  --input "$R2_ROOT/eval/denovo/candidates.csv" --output "$R2_ROOT/eval/denovo/candidates.normalized.csv"
mv "$R2_ROOT/eval/denovo/candidates.normalized.csv" "$R2_ROOT/eval/denovo/candidates.csv"
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$DENOVO_CSV" --candidates-csv "$R2_ROOT/eval/denovo/candidates.csv" \
  --output-json "$R2_ROOT/eval/denovo/metrics.json" --output-md "$R2_ROOT/eval/denovo/report.md" \
  --budgets 1,8,20
"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" sample \
  --checkpoint "$CHECKPOINT" --eval-csv "$EDIT_CSV" --eval-features-dir "$EDIT_FEATURES" \
  --candidate-output-csv "$R2_ROOT/eval/edit/candidates.csv" \
  --summary-json "$R2_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --edit-mask-fraction "$R2_FRACTION" --steps 4 --temperature 0.8 --seed 2807 --device auto
for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EDIT_CSV" --candidates "$R2_ROOT/eval/edit/candidates.csv" \
    --output-dir "$R2_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_8_r2_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EDIT_CSV" --candidates "$R2_ROOT/eval/edit/candidates.csv" \
  --output-dir "$R2_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p8_1_8_r2_candidate20 --task-filter table1 --missing-oracle-policy fail
"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" audit --checkpoint "$CHECKPOINT" \
  --summaries "$R2_ROOT/eval/denovo/sampling_summary.json" "$R2_ROOT/eval/edit/sampling_summary.json" \
  --output "$R2_ROOT/unified_audit.json"
touch "$R2_ROOT/COMPLETE"
echo "P8.1.8-R2 complete: $R2_ROOT"
