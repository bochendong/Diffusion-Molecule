#!/usr/bin/env bash
# Representation kill test: molecular graph -> atom/bond latent slots -> raw graph.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PYTHON_BIN="${SUCC_GRAPH_LATENT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_GRAPH_LATENT_SEED:-1723}"
OUTPUT_DIR="${SUCC_GRAPH_LATENT_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/graph_latent_autoencoder_v1/seed_${SEED}}"
DATASET_DIR="${SUCC_GRAPH_LATENT_DATASET_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2/dataset}"

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/train_graph_latent_autoencoder.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_GRAPH_LATENT_TRAIN_LIMIT:-15000}" \
  --validation-limit "${SUCC_GRAPH_LATENT_VALIDATION_LIMIT:-400}" \
  --max-atoms "${SUCC_GRAPH_LATENT_MAX_ATOMS:-64}" \
  --epochs "${SUCC_GRAPH_LATENT_EPOCHS:-6}" \
  --batch-size "${SUCC_GRAPH_LATENT_BATCH_SIZE:-32}" \
  --node-dim "${SUCC_GRAPH_LATENT_NODE_DIM:-192}" \
  --edge-dim "${SUCC_GRAPH_LATENT_EDGE_DIM:-64}" \
  --layers "${SUCC_GRAPH_LATENT_LAYERS:-4}" \
  --latent-noise "${SUCC_GRAPH_LATENT_NOISE:-0.05}" \
  --seed "$SEED" \
  --device auto
