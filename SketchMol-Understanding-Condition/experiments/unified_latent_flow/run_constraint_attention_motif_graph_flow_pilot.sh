#!/usr/bin/env bash
# Matched B12 ablation: source-node cross-attention over constraint tokens.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_CONSTRAINT_ATTN_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_CONSTRAINT_ATTN_SEED:-1741}"
OUTPUT_DIR="${SUCC_CONSTRAINT_ATTN_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/constraint_attention_motif_graph_flow_v12/seed_${SEED}}"
DATASET_DIR="${SUCC_CONSTRAINT_ATTN_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_CONSTRAINT_ATTN_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B12 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_CONSTRAINT_ATTN_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed B12 run exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/hierarchical_vq_motif_graph_flow.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_CONSTRAINT_ATTN_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_CONSTRAINT_ATTN_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_CONSTRAINT_ATTN_PROPERTY_COUNTS:-2,3}" \
  --constraint-codebook-size 16 \
  --constraint-code-dim 32 \
  --motif-codebook-size 64 \
  --motif-code-dim 64 \
  --epochs "${SUCC_CONSTRAINT_ATTN_EPOCHS:-8}" \
  --batch-size "${SUCC_CONSTRAINT_ATTN_BATCH_SIZE:-6}" \
  --hidden-dim 256 \
  --sampling-temperature 0.80 \
  --edit-gate-loss-weight 0.50 \
  --delta-loss-weight 0.50 \
  --valence-budget-loss-weight 0.25 \
  --motif-atom-count-loss-weight 0.25 \
  --contrastive-loss-weight 0.20 \
  --contrastive-margin 0.20 \
  --source-anchored \
  --connected-region \
  --categorical-delta \
  --valence-budget \
  --motif-attachment \
  --condition-attention \
  --condition-attention-heads "${SUCC_CONSTRAINT_ATTN_HEADS:-4}" \
  --num-attempts 20 \
  --sample-batch-size 5 \
  --seed "$SEED" \
  --device auto
