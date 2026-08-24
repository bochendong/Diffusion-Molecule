#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
P7_DIR="$PROJECT_DIR/experiments/p7_state_constrained_molprogram"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P8_SEED:-7}"
P6_ROOT="${P8_P6_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
OUTPUT_ROOT="${P8_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p8_validity_preserving_molprogram_v1/seed_${SEED}}"
DIRECT_ROOT="${P8_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
CHECKPOINT="$P6_ROOT/policy/umtp_graph_action_policy.pt"
TRAIN_CSV="$P6_ROOT/data/train_transition_programs.csv"
DENOVO_CSV="$P6_ROOT/data/denovo_hard_gate.csv"
DENOVO_FEATURES="$DIRECT_ROOT/eval_condition_features_hf_vlm"

mkdir -p "$OUTPUT_ROOT/eval/denovo"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" "$P7_DIR/p7_state_constrained_decode.py" \
  --checkpoint "$CHECKPOINT" --train-csv "$TRAIN_CSV" \
  --eval-csv "$DENOVO_CSV" --eval-features-dir "$DENOVO_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" \
  --num-samples 20 --max-new-tokens 188 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --validity-preserving --seed 1807 --device auto

"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$DENOVO_CSV" --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" \
  --output-md "$OUTPUT_ROOT/eval/denovo/report.md" --budgets 1,8,20

touch "$OUTPUT_ROOT/COMPLETE"
echo "P8 validity-preserving de-novo gate complete: $OUTPUT_ROOT"
