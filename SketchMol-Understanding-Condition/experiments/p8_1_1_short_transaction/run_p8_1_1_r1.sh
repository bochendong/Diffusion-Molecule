#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
UNIFIED_DIR="$PROJECT_DIR/experiments/unified_smiles_generator"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P811_SEED:-7}"
OUTPUT_ROOT="${P811_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p8_1_1_short_transaction_r1/seed_${SEED}}"
P1_CHECKPOINT="${P811_P1_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
V2_ROOT="${P811_V2_ROOT:-$PROJECT_DIR/outputs/umtp_graph_action_instruction_v2/seed_${SEED}}"
P6_ROOT="${P811_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
DIRECT_ROOT="${P811_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
SUITE_ROOT="${P811_SUITE_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
JOINT_ROOT="${P811_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
TRAIN_INPUT="$V2_ROOT/data/action_train_instruction_v2.csv"
VALIDATION_INPUT="$V2_ROOT/data/action_validation_instruction_v2.csv"
EDIT_CSV="$P6_ROOT/data/edit_table1_gate.csv"
DENOVO_CSV="$P6_ROOT/data/denovo_hard_gate.csv"
TRAIN_FEATURES="$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"
DENOVO_FEATURES="$DIRECT_ROOT/eval_condition_features_hf_vlm"
TRAIN_CSV="$OUTPUT_ROOT/data/edit_train.csv"
VALIDATION_CSV="$OUTPUT_ROOT/data/edit_validation.csv"
POLICY_DIR="$OUTPUT_ROOT/policy"
CHECKPOINT="$POLICY_DIR/umtp_graph_action_policy.pt"

for path in "$P1_CHECKPOINT" "$TRAIN_INPUT" "$VALIDATION_INPUT" "$EDIT_CSV" "$DENOVO_CSV"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P8.1.1 input: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$EDIT_FEATURES" "$DENOVO_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing P8.1.1 features: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT/data" "$POLICY_DIR" "$OUTPUT_ROOT/eval/denovo" "$OUTPUT_ROOT/eval/edit"
export PYTHONPATH="$PROJECT_DIR:$UNIFIED_DIR:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" "$SCRIPT_DIR/filter_edit_rows.py" --input "$TRAIN_INPUT" --output "$TRAIN_CSV"
"$PYTHON_BIN" "$SCRIPT_DIR/filter_edit_rows.py" --input "$VALIDATION_INPUT" --output "$VALIDATION_CSV"

echo "=== P8.1.1-R1 train only the short source transaction path ==="
"$PYTHON_BIN" "$UNIFIED_DIR/umtp_graph_action_policy.py" train \
  --base-checkpoint "$P1_CHECKPOINT" --source-aware-warmstart \
  --train-csv "$TRAIN_CSV" --eval-csv "$VALIDATION_CSV" \
  --train-features-dir "$TRAIN_FEATURES" --eval-features-dir "$EDIT_FEATURES" \
  --output-dir "$POLICY_DIR" --condition-layout direct_compat \
  --max-smiles-length 160 --epochs "${P811_EPOCHS:-4}" --batch-size 64 --eval-batch-size 128 \
  --samples-per-epoch "${P811_SAMPLES_PER_EPOCH:-4096}" --lr "${P811_LR:-8e-5}" \
  --weight-decay 0 --distill-weight 0 --trainable-scope source_action --seed "$SEED" --device auto

"$PYTHON_BIN" "$SCRIPT_DIR/audit_unified_checkpoint.py" \
  --base-checkpoint "$P1_CHECKPOINT" --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_CSV" \
  --denovo-eval-csv "$DENOVO_CSV" --edit-eval-csv "$EDIT_CSV" \
  --output "$OUTPUT_ROOT/pre_eval_audit.json"

echo "=== P8.1.1-R1 raw de-novo samples ==="
"$PYTHON_BIN" "$SCRIPT_DIR/sample_raw_denovo.py" \
  --checkpoint "$CHECKPOINT" --eval-csv "$DENOVO_CSV" --eval-features-dir "$DENOVO_FEATURES" \
  --output-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" --num-samples 20 --seed 1907 --device auto
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$DENOVO_CSV" --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" \
  --output-md "$OUTPUT_ROOT/eval/denovo/report.md" --budgets 1,8,20

echo "=== P8.1.1-R1 raw source-only transaction samples ==="
"$PYTHON_BIN" "$SCRIPT_DIR/sample_raw_transactions.py" \
  --checkpoint "$CHECKPOINT" --eval-csv "$EDIT_CSV" --eval-features-dir "$EDIT_FEATURES" \
  --output-csv "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --temperature "${P811_TEMPERATURE:-1.0}" --seed 2907 --device auto

for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_1_r1_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p8_1_1_r1_candidate20 --task-filter table1 --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/audit_unified_checkpoint.py" \
  --base-checkpoint "$P1_CHECKPOINT" --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_CSV" \
  --denovo-eval-csv "$DENOVO_CSV" --edit-eval-csv "$EDIT_CSV" \
  --denovo-summary "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" \
  --edit-summary "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --output "$OUTPUT_ROOT/final_audit.json"
touch "$OUTPUT_ROOT/COMPLETE"
echo "P8.1.1-R1 complete: $OUTPUT_ROOT"

