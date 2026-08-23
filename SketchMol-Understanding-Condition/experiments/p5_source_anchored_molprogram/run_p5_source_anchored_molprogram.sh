#!/usr/bin/env bash
# P5 single-seed source-anchored pointer-generator gate.

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
SEED="${P5_SEED:-7}"
OUTPUT_ROOT="${P5_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p5_source_anchored_molprogram_v1/seed_${SEED}}"
P4_ROOT="${P5_P4_ROOT:-$PROJECT_DIR/outputs/p4_event_to_smiles_distillation_v1_h100/seed_${SEED}}"
JOINT_ROOT="${P5_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${P5_SUITE_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
BASE_CHECKPOINT="${P5_BASE_CHECKPOINT:-$PROJECT_DIR/outputs/unified_molecular_transformation_policy_v1/seed_${SEED}/policy/unified_smiles_generator.pt}"
DISTILL_ROWS="${P5_DISTILL_ROWS:-$P4_ROOT/data/distillation_rows.csv}"
VALIDATION_SOURCE="${P5_VALIDATION_SOURCE:-$PROJECT_DIR/outputs/umtp_graph_action_protected_pilot_v1/seed_${SEED}/data/table1_pool.csv}"
TRAIN_FEATURES="${P5_TRAIN_FEATURES:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
VALIDATION_FEATURES="${P5_VALIDATION_FEATURES:-$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm}"
PREREG="$SCRIPT_DIR/p5_preregistration.json"
SFT_DIR="$OUTPUT_ROOT/student_sft"
GRPO_DIR="$OUTPUT_ROOT/student_grpo"
SFT_CHECKPOINT="$SFT_DIR/unified_smiles_generator.pt"
GRPO_CHECKPOINT="$GRPO_DIR/unified_smiles_generator_group_rl.pt"

if [[ -f "$OUTPUT_ROOT/p5_summary.json" && "${P5_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed P5 output exists: $OUTPUT_ROOT/p5_summary.json" >&2
  exit 2
fi
for path in "$BASE_CHECKPOINT" "$DISTILL_ROWS" "$VALIDATION_SOURCE" "$PREREG"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P5 input: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$VALIDATION_FEATURES" "$P4_ROOT/eval/base"; do
  [[ -d "$path" ]] || { echo "ERROR: missing P5 input directory: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT/eval" "$SFT_DIR" "$GRPO_DIR"
export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts:$PROJECT_DIR/experiments/unified_latent_flow${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== P5 source-anchored SFT ==="
"$PYTHON_BIN" "$UNIFIED_DIR/unified_smiles_generator.py" train \
  --train-csv "$DISTILL_ROWS" \
  --condition-features-dir "$TRAIN_FEATURES" \
  --condition-layout transformation \
  --resume-checkpoint "$BASE_CHECKPOINT" \
  --reset-training-state \
  --allow-architecture-warmstart \
  --source-copy-aware \
  --source-adapter-layers 2 \
  --source-adapter-bottleneck 64 \
  --source-copy-initial-vocab-bias 2.0 \
  --trainable-scope source_only \
  --output-dir "$SFT_DIR" \
  --epochs "${P5_SFT_EPOCHS:-4}" \
  --batch-size "${P5_SFT_BATCH_SIZE:-32}" \
  --lr "${P5_SFT_LR:-1e-4}" \
  --weight-decay 0 \
  --grad-clip 1.0 \
  --seed "$SEED" \
  --device auto

echo "=== P5 source-only GRPO ==="
"$PYTHON_BIN" "$UNIFIED_DIR/unified_smiles_generator.py" group-rl \
  --train-csv "$DISTILL_ROWS" \
  --condition-features-dir "$TRAIN_FEATURES" \
  --condition-layout transformation \
  --resume-checkpoint "$SFT_CHECKPOINT" \
  --trainable-scope source_only \
  --output-dir "$GRPO_DIR" \
  --epochs "${P5_GRPO_EPOCHS:-1}" \
  --batch-size "${P5_GRPO_BATCH_SIZE:-4}" \
  --lr "${P5_GRPO_LR:-2e-6}" \
  --weight-decay 0 \
  --rollouts-per-prompt "${P5_GRPO_ROLLOUTS:-8}" \
  --rl-objective grpo \
  --grpo-clip-eps 0.2 \
  --grpo-update-epochs 1 \
  --sft-weight 1.0 \
  --reference-kl-weight 0.05 \
  --reward-mode table1_edit \
  --reward-aggregation dense_softmin \
  --reward-valid-weight 0.5 \
  --reward-strict-weight 2.0 \
  --reward-source-similarity-weight 2.0 \
  --reward-source-similarity-threshold 0.65 \
  --reward-source-copy-penalty 0.25 \
  --smiles-grammar-constraint \
  --max-new-tokens 120 \
  --temperature 0.80 \
  --top-k 40 \
  --top-p 0.95 \
  --parallel-samples 4 \
  --max-parallel-sequences 256 \
  --seed "$SEED" \
  --device auto

"$PYTHON_BIN" "$SCRIPT_DIR/audit_checkpoint_protection.py" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --trained-checkpoint "$SFT_CHECKPOINT" \
  --output-json "$OUTPUT_ROOT/sft_checkpoint_audit.json"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_checkpoint_protection.py" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --trained-checkpoint "$GRPO_CHECKPOINT" \
  --output-json "$OUTPUT_ROOT/grpo_checkpoint_audit.json"

sample_variant() {
  local variant="$1"
  local checkpoint="$2"
  local variant_dir="$OUTPUT_ROOT/eval/$variant"
  mkdir -p "$variant_dir/candidates"
  "$PYTHON_BIN" "$UNIFIED_DIR/unified_smiles_generator.py" sample \
    --checkpoint "$checkpoint" \
    --eval-csv "$VALIDATION_SOURCE" \
    --eval-condition-features-dir "$VALIDATION_FEATURES" \
    --condition-layout transformation \
    --output-dir "$variant_dir/candidates" \
    --prediction-csv "$variant_dir/selected_raw.csv" \
    --candidate-output-csv "$variant_dir/candidates.csv" \
    --method "p5_${variant}" \
    --decoding-mode sample \
    --num-samples 20 \
    --top-k-candidates 20 \
    --max-candidates 20 \
    --disable-finalizer \
    --smiles-grammar-constraint \
    --max-new-tokens 120 \
    --temperature 0.80 \
    --top-k 40 \
    --top-p 0.95 \
    --parallel-samples 4 \
    --max-parallel-sequences 256 \
    --seed 1407 \
    --device auto
  for budget in 1 8 20; do
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
      --reference "$VALIDATION_SOURCE" \
      --candidates "$variant_dir/candidates.csv" \
      --output-dir "$variant_dir/any${budget}" \
      --candidate-limit "$budget" \
      --model-name "p5_${variant}_any${budget}" \
      --task-filter table1 \
      --missing-oracle-policy fail
  done
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$VALIDATION_SOURCE" \
    --candidates "$variant_dir/candidates.csv" \
    --output-dir "$variant_dir/candidate20" \
    --candidate-limit 20 \
    --aggregation candidate \
    --model-name "p5_${variant}_n20" \
    --task-filter table1 \
    --missing-oracle-policy fail
}

echo "=== P5 honest raw and n=20 evaluation ==="
sample_variant sft "$SFT_CHECKPOINT"
sample_variant grpo "$GRPO_CHECKPOINT"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_p5.py" \
  --output-root "$OUTPUT_ROOT" \
  --preregistration "$PREREG" \
  --base-eval-root "$P4_ROOT/eval" \
  --sft-audit "$OUTPUT_ROOT/sft_checkpoint_audit.json" \
  --grpo-audit "$OUTPUT_ROOT/grpo_checkpoint_audit.json"

echo "P5 ready: $OUTPUT_ROOT/p5_report.md"

