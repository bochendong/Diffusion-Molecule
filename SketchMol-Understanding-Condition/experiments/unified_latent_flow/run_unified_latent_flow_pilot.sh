#!/usr/bin/env bash
# Run the direct-decoding language-conditioned molecular latent-flow kill test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$CODE_REPO_DIR/SketchMol-Understanding-Condition"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_LATENT_FLOW_SEED:-1715}"
OUTPUT_DIR="${SUCC_LATENT_FLOW_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/unified_latent_flow_pilot_v1/seed_${SEED}}"

TRAIN_CSV="${SUCC_LATENT_FLOW_TRAIN_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/unified_joint_train_rows.csv}"
VALIDATION_CSV="${SUCC_LATENT_FLOW_VALIDATION_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/unified_joint_validation_rows.csv}"
TRAIN_FEATURES="${SUCC_LATENT_FLOW_TRAIN_FEATURES:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/feature_variants/train_condition_features_hf_vlm}"
VALIDATION_FEATURES="${SUCC_LATENT_FLOW_VALIDATION_FEATURES:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/feature_variants/validation_condition_features_hf_vlm}"
BASE_CHECKPOINT="${SUCC_LATENT_FLOW_BASE_CHECKPOINT:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/u2_joint_protected_sft/seed_7/unified_smiles_generator.pt}"

for path in "$TRAIN_CSV" "$VALIDATION_CSV" "$TRAIN_FEATURES/query_tokens.npy" \
  "$VALIDATION_FEATURES/query_tokens.npy" "$BASE_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required latent-flow input: $path" >&2
    exit 2
  fi
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_LATENT_FLOW_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed latent-flow pilot exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$SCRIPT_DIR/unified_latent_flow.py" \
  --train-csv "$TRAIN_CSV" \
  --validation-csv "$VALIDATION_CSV" \
  --train-features-dir "$TRAIN_FEATURES" \
  --validation-features-dir "$VALIDATION_FEATURES" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --feature-variant full \
  --train-limit "${SUCC_LATENT_FLOW_TRAIN_LIMIT:-12000}" \
  --validation-per-mode "${SUCC_LATENT_FLOW_VALIDATION_PER_MODE:-25}" \
  --epochs "${SUCC_LATENT_FLOW_EPOCHS:-1}" \
  --batch-size "${SUCC_LATENT_FLOW_BATCH_SIZE:-48}" \
  --learning-rate "${SUCC_LATENT_FLOW_LR:-2e-4}" \
  --latent-tokens "${SUCC_LATENT_FLOW_TOKENS:-8}" \
  --encoder-layers "${SUCC_LATENT_FLOW_ENCODER_LAYERS:-2}" \
  --flow-layers "${SUCC_LATENT_FLOW_LAYERS:-3}" \
  --flow-steps "${SUCC_LATENT_FLOW_STEPS:-8}" \
  --decoder-corruption "${SUCC_LATENT_FLOW_DECODER_CORRUPTION:-0.30}" \
  --num-samples 20 \
  --sample-batch-size 20 \
  --seed "$SEED" \
  --device cuda

echo "Latent-flow pilot ready: $OUTPUT_DIR/summary.json"
