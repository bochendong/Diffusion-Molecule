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
OUTPUT_ROOT="${P818_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p8_1_8_masked_molecule_policy_r1/seed_${SEED}}"
V2_ROOT="${P818_V2_ROOT:-$PROJECT_DIR/outputs/umtp_graph_action_instruction_v2/seed_${SEED}}"
P6_ROOT="${P818_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
DIRECT_ROOT="${P818_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
SUITE_ROOT="${P818_SUITE_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
JOINT_ROOT="${P818_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
TRAIN_CSV="$V2_ROOT/data/action_train_instruction_v2.csv"
DENOVO_CSV="$P6_ROOT/data/denovo_hard_gate.csv"
EDIT_CSV="$P6_ROOT/data/edit_table1_gate.csv"
TRAIN_FEATURES="$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm"
DENOVO_FEATURES="$DIRECT_ROOT/eval_condition_features_hf_vlm"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"
CHECKPOINT="$OUTPUT_ROOT/policy/masked_molecule_policy.pt"

for path in "$TRAIN_CSV" "$DENOVO_CSV" "$EDIT_CSV"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P8.1.8 input: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$DENOVO_FEATURES" "$EDIT_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing P8.1.8 features: $path" >&2; exit 2; }
done
mkdir -p "$OUTPUT_ROOT/policy" "$OUTPUT_ROOT/eval/denovo" "$OUTPUT_ROOT/eval/edit"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" preflight \
  --train-csv "$TRAIN_CSV" --denovo-eval-csv "$DENOVO_CSV" --edit-eval-csv "$EDIT_CSV" \
  --max-tokens "${P818_MAX_TOKENS:-80}" --output "$OUTPUT_ROOT/representation_preflight.json"

"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" train \
  --train-csv "$TRAIN_CSV" --train-features-dir "$TRAIN_FEATURES" --output-dir "$OUTPUT_ROOT/policy" \
  --max-tokens "${P818_MAX_TOKENS:-80}" --epochs "${P818_EPOCHS:-10}" \
  --batch-size "${P818_BATCH_SIZE:-64}" --edit-mask-fraction 0.35 --seed "$SEED" --device auto

"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" sample \
  --checkpoint "$CHECKPOINT" --eval-csv "$DENOVO_CSV" --eval-features-dir "$DENOVO_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" \
  --num-samples 20 --edit-mask-fraction 0.35 --steps 4 --temperature 0.8 --seed 1807 --device auto
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$DENOVO_CSV" --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" --output-md "$OUTPUT_ROOT/eval/denovo/report.md" \
  --budgets 1,8,20

"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" sample \
  --checkpoint "$CHECKPOINT" --eval-csv "$EDIT_CSV" --eval-features-dir "$EDIT_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --edit-mask-fraction 0.35 --steps 4 --temperature 0.8 --seed 2807 --device auto
for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_8_r1_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p8_1_8_r1_candidate20 --task-filter table1 --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/masked_molecule_policy.py" audit --checkpoint "$CHECKPOINT" \
  --summaries "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --output "$OUTPUT_ROOT/unified_audit.json"
touch "$OUTPUT_ROOT/COMPLETE"
echo "P8.1.8-R1 complete: $OUTPUT_ROOT"

