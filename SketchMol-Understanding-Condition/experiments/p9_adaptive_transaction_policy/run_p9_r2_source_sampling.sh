#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
P811_DIR="$PROJECT_DIR/experiments/p8_1_1_short_transaction"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P9_SEED:-7}"
R1_ROOT="${P9_R1_ROOT:-$PROJECT_DIR/outputs/p9_adaptive_transaction_policy_v1/seed_${SEED}}"
OUTPUT_ROOT="${P9_R2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p9_adaptive_transaction_policy_r2_source_sampling_v1/seed_${SEED}}"
P6_ROOT="${P9_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
JOINT_ROOT="${P9_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
CHECKPOINT="$R1_ROOT/policy/umtp_graph_action_policy.pt"
EDIT_CSV="$P6_ROOT/data/edit_table1_gate.csv"
EDIT_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"

for path in "$CHECKPOINT" "$EDIT_CSV"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P9-R2 input: $path" >&2; exit 2; }
done
[[ -d "$EDIT_FEATURES" ]] || { echo "ERROR: missing P9-R2 features: $EDIT_FEATURES" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT/eval/edit"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

echo "P9-R2 sole change: instruction-enumerated ranking -> raw source-only transaction sampling."
"$PYTHON_BIN" "$P811_DIR/sample_raw_transactions.py" \
  --checkpoint "$CHECKPOINT" --eval-csv "$EDIT_CSV" --eval-features-dir "$EDIT_FEATURES" \
  --output-csv "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --num-samples 20 --temperature "${P9_R2_TEMPERATURE:-1.0}" --seed 2907 --device auto

for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" --candidate-limit "$budget" \
    --model-name "p9_r2_source_sampling_any${budget}" --task-filter table1 \
    --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EDIT_CSV" --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate \
  --model-name p9_r2_source_sampling_candidate20 --task-filter table1 --missing-oracle-policy fail

cat > "$OUTPUT_ROOT/preregistration.txt" <<'EOF'
R1: instruction-conditioned grammar enumeration followed by checkpoint ranking.
R2 sole scientific change: direct source-only transaction sampling from the same checkpoint.
Held fixed: seed, checkpoint, edit subset, condition features, raw candidate count, k=1/8/20 evaluators, and no property reranking.
EOF
touch "$OUTPUT_ROOT/COMPLETE"
echo "P9-R2 source-only sampling complete: $OUTPUT_ROOT"
