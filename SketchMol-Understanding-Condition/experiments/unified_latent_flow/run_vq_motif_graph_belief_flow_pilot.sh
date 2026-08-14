#!/usr/bin/env bash
# Single-token VQ motif graph-belief pilot, 2p+3p, exact n=20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_VQ_MOTIF_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_VQ_MOTIF_SEED:-1739}"
OUTPUT_DIR="${SUCC_VQ_MOTIF_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/vq_motif_graph_belief_flow_v5b/seed_${SEED}}"
DATASET_DIR="${SUCC_VQ_MOTIF_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_VQ_MOTIF_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing VQ-motif input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_VQ_MOTIF_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed VQ-motif pilot exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/vq_motif_graph_belief_flow.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_VQ_MOTIF_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_VQ_MOTIF_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_VQ_MOTIF_PROPERTY_COUNTS:-2,3}" \
  --codebook-size "${SUCC_VQ_MOTIF_CODEBOOK_SIZE:-64}" \
  --code-dim "${SUCC_VQ_MOTIF_CODE_DIM:-64}" \
  --epochs "${SUCC_VQ_MOTIF_EPOCHS:-8}" \
  --batch-size "${SUCC_VQ_MOTIF_BATCH_SIZE:-8}" \
  --hidden-dim "${SUCC_VQ_MOTIF_HIDDEN_DIM:-256}" \
  --sampling-temperature "${SUCC_VQ_MOTIF_TEMPERATURE:-0.80}" \
  --contrastive-loss-weight "${SUCC_VQ_MOTIF_CONTRASTIVE_WEIGHT:-0.25}" \
  --contrastive-margin "${SUCC_VQ_MOTIF_CONTRASTIVE_MARGIN:-0.20}" \
  --gate-min-active-codes "${SUCC_VQ_MOTIF_MIN_ACTIVE_CODES:-4}" \
  --num-attempts 20 \
  --sample-batch-size "${SUCC_VQ_MOTIF_SAMPLE_BATCH_SIZE:-5}" \
  --seed "$SEED" \
  --device auto
