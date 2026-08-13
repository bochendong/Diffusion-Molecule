#!/usr/bin/env bash
# Representation-only stage: molecule -> latent tokens -> molecule.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PYTHON_BIN="${SUCC_LATENT_FLOW_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_MOL_AE_SEED:-1717}"
OUTPUT_DIR="${SUCC_MOL_AE_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/molecular_latent_autoencoder_v2/seed_${SEED}}"
DATASET_DIR="${SUCC_MOL_AE_DATASET_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2/dataset}"
BASE_CHECKPOINT="${SUCC_MOL_AE_BASE_CHECKPOINT:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2/u2_joint_protected_sft/seed_7/unified_smiles_generator.pt}"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/train_molecular_latent_autoencoder.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_MOL_AE_TRAIN_LIMIT:-30000}" \
  --validation-limit "${SUCC_MOL_AE_VALIDATION_LIMIT:-400}" \
  --epochs "${SUCC_MOL_AE_EPOCHS:-4}" \
  --batch-size "${SUCC_MOL_AE_BATCH_SIZE:-64}" \
  --latent-tokens "${SUCC_MOL_AE_LATENT_TOKENS:-16}" \
  --encoder-layers "${SUCC_MOL_AE_ENCODER_LAYERS:-3}" \
  --decoder-corruption "${SUCC_MOL_AE_DECODER_CORRUPTION:-0.40}" \
  --latent-noise "${SUCC_MOL_AE_LATENT_NOISE:-0.03}" \
  --latent-swap-weight "${SUCC_MOL_AE_SWAP_WEIGHT:-0.25}" \
  --geometry-weight "${SUCC_MOL_AE_GEOMETRY_WEIGHT:-0.10}" \
  --seed "$SEED" \
  --device auto
