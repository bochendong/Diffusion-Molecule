#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
P6_DIR="$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P812_SEED:-7}"
OUTPUT_ROOT="${P812_OUTPUT_ROOT:?set P812_OUTPUT_ROOT to the completed training root}"
ORACLE_ROOT="${P812_ORACLE_ROOT:-$PROJECT_DIR/outputs/p8_1_2_unified_transduction_oracle/seed_${SEED}}"
P6_ROOT="${P812_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
DIRECT_ROOT="${P812_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
JOINT_ROOT="${P812_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
CHECKPOINT="$OUTPUT_ROOT/policy/umtp_graph_action_policy.pt"
TRAIN_R2="$ORACLE_ROOT/r2/transduction_rows.csv"
DENOVO_FEATURES="$DIRECT_ROOT/eval_condition_features_hf_vlm"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"

for path in "$CHECKPOINT" "$TRAIN_R2" "$P6_ROOT/data/denovo_hard_gate.csv" "$P6_ROOT/data/edit_table1_gate.csv"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P8.1.2 eval input: $path" >&2; exit 2; }
done
for path in "$DENOVO_FEATURES" "$EDIT_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing P8.1.2 feature store: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT/eval/denovo" "$OUTPUT_ROOT/eval/edit"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

echo "Implementation-only continuation: reuse trained checkpoint and filter grammar terminals to its sealed vocabulary."
"$PYTHON_BIN" "$SCRIPT_DIR/sample_transduction_policy.py" \
  --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_R2" \
  --eval-csv "$P6_ROOT/data/denovo_hard_gate.csv" --eval-features-dir "$DENOVO_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" \
  --num-samples 20 --max-new-tokens 128 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --seed 1812 --device auto
"$PYTHON_BIN" "$SCRIPT_DIR/sample_transduction_policy.py" \
  --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_R2" \
  --eval-csv "$P6_ROOT/data/edit_table1_gate.csv" --eval-features-dir "$EDIT_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --max-new-tokens 64 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --seed 2812 --device auto

"$PYTHON_BIN" "$P6_DIR/evaluate_p6_denovo_gate.py" \
  --eval-csv "$P6_ROOT/data/denovo_hard_gate.csv" \
  --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" \
  --output-md "$OUTPUT_ROOT/eval/denovo/report.md" --budgets 1,8,20
for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$P6_ROOT/data/edit_table1_gate.csv" \
    --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p8_1_2_raw_any${budget}" --task-filter table1 --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$P6_ROOT/data/edit_table1_gate.csv" \
  --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p8_1_2_raw_candidate20 --task-filter table1 --missing-oracle-policy fail
touch "$OUTPUT_ROOT/COMPLETE"
echo "P8.1.2 eval-only continuation complete: $OUTPUT_ROOT"
