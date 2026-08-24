#!/usr/bin/env bash
# Fast frozen-checkpoint gate: one state-constrained decoder for both modes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P7_SEED:-7}"
P6_ROOT="${P7_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
OUTPUT_ROOT="${P7_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p7_state_constrained_molprogram_v1/seed_${SEED}}"
DIRECT_ROOT="${P7_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
JOINT_ROOT="${P7_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
CHECKPOINT="$P6_ROOT/policy/umtp_graph_action_policy.pt"
TRAIN_CSV="$P6_ROOT/data/train_transition_programs.csv"
DENOVO_CSV="$P6_ROOT/data/denovo_hard_gate.csv"
EDIT_CSV="$P6_ROOT/data/edit_table1_gate.csv"
DENOVO_FEATURES="$DIRECT_ROOT/eval_condition_features_hf_vlm"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"

for path in "$CHECKPOINT" "$TRAIN_CSV" "$DENOVO_CSV" "$EDIT_CSV"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P7 input: $path" >&2; exit 2; }
done
for path in "$DENOVO_FEATURES" "$EDIT_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing P7 feature directory: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT/eval/denovo" "$OUTPUT_ROOT/eval/edit"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

echo "=== P7 frozen P6 checkpoint, same constrained decoder in both modes ==="
"$PYTHON_BIN" "$SCRIPT_DIR/p7_state_constrained_decode.py" \
  --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_CSV" \
  --eval-csv "$DENOVO_CSV" --eval-features-dir "$DENOVO_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" \
  --num-samples 20 --max-new-tokens 188 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --seed 1707 --device auto

"$PYTHON_BIN" "$SCRIPT_DIR/p7_state_constrained_decode.py" \
  --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_CSV" \
  --eval-csv "$EDIT_CSV" --eval-features-dir "$EDIT_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --max-new-tokens 16 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --seed 2707 --device auto

"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$DENOVO_CSV" --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" \
  --output-md "$OUTPUT_ROOT/eval/denovo/report.md" --budgets 1,8,20

for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p7_state_constrained_any${budget}" --task-filter table1 \
    --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p7_state_constrained_candidate20 --task-filter table1 --missing-oracle-policy fail

touch "$OUTPUT_ROOT/COMPLETE"
echo "P7 state-constrained gate complete: $OUTPUT_ROOT"
