#!/usr/bin/env bash
# Fast single-seed gate for one empty/source graph transition policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
UNIFIED_DIR="$PROJECT_DIR/experiments/unified_smiles_generator"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P6_SEED:-7}"
OUTPUT_ROOT="${P6_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}}"
V2_ROOT="${P6_V2_ROOT:-$PROJECT_DIR/outputs/umtp_graph_action_instruction_v2/seed_${SEED}}"
DIRECT_ROOT="${P6_DIRECT_ROOT:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
SUITE_ROOT="${P6_SUITE_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
JOINT_ROOT="${P6_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
BASE_CHECKPOINT="${P6_BASE_CHECKPOINT:-$V2_ROOT/action_policy/umtp_graph_action_policy.pt}"
SOURCE_TRAIN="$V2_ROOT/data/action_train_instruction_v2.csv"
SOURCE_VALIDATION="$V2_ROOT/data/action_validation_instruction_v2.csv"
TABLE1_SOURCE="$V2_ROOT/data/table1_pool.csv"
DENOVO_SOURCE="$DIRECT_ROOT/denovo_2p7p_eval_rows.csv"
TRAIN_FEATURES="$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm"
VALIDATION_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"
DENOVO_FEATURES="$DIRECT_ROOT/eval_condition_features_hf_vlm"
MODEL_DIR="$OUTPUT_ROOT/policy"
CHECKPOINT="$MODEL_DIR/umtp_graph_action_policy.pt"

if [[ -f "$OUTPUT_ROOT/COMPLETE" && "${P6_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed P6 gate exists: $OUTPUT_ROOT" >&2
  exit 2
fi
for path in "$BASE_CHECKPOINT" "$SOURCE_TRAIN" "$SOURCE_VALIDATION" "$TABLE1_SOURCE" "$DENOVO_SOURCE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P6 input: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$VALIDATION_FEATURES" "$DENOVO_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing P6 feature directory: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT/data" "$OUTPUT_ROOT/eval/denovo" "$OUTPUT_ROOT/eval/edit" "$MODEL_DIR"
export PYTHONPATH="$PROJECT_DIR:$UNIFIED_DIR:$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== P6 build one transition language for both task modes ==="
"$PYTHON_BIN" "$SCRIPT_DIR/p6_transition_program.py" prepare \
  --input-csv "$SOURCE_TRAIN" \
  --output-csv "$OUTPUT_ROOT/data/train_transition_programs.csv" \
  --manifest-json "$OUTPUT_ROOT/data/train_transition_programs.manifest.json" \
  --max-program-tokens 188
"$PYTHON_BIN" "$SCRIPT_DIR/p6_transition_program.py" prepare \
  --input-csv "$SOURCE_VALIDATION" \
  --output-csv "$OUTPUT_ROOT/data/validation_transition_programs.csv" \
  --manifest-json "$OUTPUT_ROOT/data/validation_transition_programs.manifest.json" \
  --max-program-tokens 188

echo "=== P6 train exactly one decoder/checkpoint ==="
"$PYTHON_BIN" "$UNIFIED_DIR/umtp_graph_action_policy.py" train \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --train-csv "$OUTPUT_ROOT/data/train_transition_programs.csv" \
  --eval-csv "$OUTPUT_ROOT/data/validation_transition_programs.csv" \
  --train-features-dir "$TRAIN_FEATURES" \
  --eval-features-dir "$VALIDATION_FEATURES" \
  --output-dir "$MODEL_DIR" \
  --condition-layout p6_transition \
  --max-smiles-length 188 \
  --epochs "${P6_EPOCHS:-4}" \
  --batch-size "${P6_BATCH_SIZE:-48}" \
  --eval-batch-size 64 \
  --samples-per-epoch "${P6_SAMPLES_PER_EPOCH:-8192}" \
  --lr "${P6_LR:-8e-5}" \
  --weight-decay 1e-4 \
  --distill-weight 0 \
  --trainable-scope all \
  --seed "$SEED" \
  --device auto

echo "=== P6 freeze bounded hard-de-novo and all-task editing gates ==="
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_p6_gate_subset.py" \
  --input-csv "$DENOVO_SOURCE" \
  --output-csv "$OUTPUT_ROOT/data/denovo_hard_gate.csv" \
  --manifest-json "$OUTPUT_ROOT/data/denovo_hard_gate.manifest.json" \
  --mode denovo_hard --rows-per-group "${P6_DENOVO_ROWS_PER_GROUP:-32}" --seed 20260824
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_p6_gate_subset.py" \
  --input-csv "$TABLE1_SOURCE" \
  --output-csv "$OUTPUT_ROOT/data/edit_table1_gate.csv" \
  --manifest-json "$OUTPUT_ROOT/data/edit_table1_gate.manifest.json" \
  --mode edit_table1 --rows-per-group "${P6_EDIT_ROWS_PER_GROUP:-20}" --seed 20260824

echo "=== P6 sample both modes through the same decoder and interpreter ==="
"$PYTHON_BIN" "$SCRIPT_DIR/p6_transition_program.py" sample \
  --checkpoint "$CHECKPOINT" \
  --eval-csv "$OUTPUT_ROOT/data/denovo_hard_gate.csv" \
  --eval-features-dir "$DENOVO_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/denovo/sampling_summary.json" \
  --condition-layout p6_transition \
  --num-samples 20 --max-new-tokens 188 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --seed 1407 --device auto
"$PYTHON_BIN" "$SCRIPT_DIR/p6_transition_program.py" sample \
  --checkpoint "$CHECKPOINT" \
  --eval-csv "$OUTPUT_ROOT/data/edit_table1_gate.csv" \
  --eval-features-dir "$VALIDATION_FEATURES" \
  --candidate-output-csv "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --summary-json "$OUTPUT_ROOT/eval/edit/sampling_summary.json" \
  --condition-layout p6_transition \
  --num-samples 20 --max-new-tokens 64 --temperature 0.8 --top-k 32 --top-p 0.95 \
  --seed 2407 --device auto

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_p6_denovo_gate.py" \
  --eval-csv "$OUTPUT_ROOT/data/denovo_hard_gate.csv" \
  --candidates-csv "$OUTPUT_ROOT/eval/denovo/candidates.csv" \
  --output-json "$OUTPUT_ROOT/eval/denovo/metrics.json" \
  --output-md "$OUTPUT_ROOT/eval/denovo/report.md" --budgets 1,8,20

for budget in 1 8 20; do
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$OUTPUT_ROOT/data/edit_table1_gate.csv" \
    --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
    --output-dir "$OUTPUT_ROOT/eval/edit/any${budget}" \
    --candidate-limit "$budget" \
    --model-name "p6_unified_transition_any${budget}" \
    --task-filter table1 \
    --missing-oracle-policy fail
done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$OUTPUT_ROOT/data/edit_table1_gate.csv" \
  --candidates "$OUTPUT_ROOT/eval/edit/candidates.csv" \
  --output-dir "$OUTPUT_ROOT/eval/edit/candidate20" \
  --candidate-limit 20 --aggregation candidate \
  --model-name p6_unified_transition_candidate20 \
  --task-filter table1 --missing-oracle-policy fail

touch "$OUTPUT_ROOT/COMPLETE"
echo "P6 unified transition gate complete: $OUTPUT_ROOT"
