#!/usr/bin/env bash
# Connected motif-attachment hierarchical VQ pilot, exact n=20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_MOTIF_ATTACH_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_MOTIF_ATTACH_SEED:-1741}"
OUTPUT_DIR="${SUCC_MOTIF_ATTACH_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/motif_attachment_hierarchical_vq_graph_flow_v11/seed_${SEED}}"
DATASET_DIR="${SUCC_MOTIF_ATTACH_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_MOTIF_ATTACH_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing motif-attachment input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_MOTIF_ATTACH_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed motif-attachment run exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/hierarchical_vq_motif_graph_flow.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_MOTIF_ATTACH_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_MOTIF_ATTACH_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_MOTIF_ATTACH_PROPERTY_COUNTS:-2,3}" \
  --constraint-codebook-size "${SUCC_MOTIF_ATTACH_CONSTRAINT_CODEBOOK_SIZE:-16}" \
  --constraint-code-dim "${SUCC_MOTIF_ATTACH_CONSTRAINT_CODE_DIM:-32}" \
  --motif-codebook-size "${SUCC_MOTIF_ATTACH_MOTIF_CODEBOOK_SIZE:-64}" \
  --motif-code-dim "${SUCC_MOTIF_ATTACH_MOTIF_CODE_DIM:-64}" \
  --epochs "${SUCC_MOTIF_ATTACH_EPOCHS:-8}" \
  --batch-size "${SUCC_MOTIF_ATTACH_BATCH_SIZE:-6}" \
  --hidden-dim "${SUCC_MOTIF_ATTACH_HIDDEN_DIM:-256}" \
  --sampling-temperature "${SUCC_MOTIF_ATTACH_TEMPERATURE:-0.80}" \
  --edit-gate-loss-weight "${SUCC_MOTIF_ATTACH_REGION_WEIGHT:-0.50}" \
  --delta-loss-weight "${SUCC_MOTIF_ATTACH_DELTA_WEIGHT:-0.50}" \
  --valence-budget-loss-weight "${SUCC_MOTIF_ATTACH_BUDGET_WEIGHT:-0.25}" \
  --motif-atom-count-loss-weight "${SUCC_MOTIF_ATTACH_COUNT_WEIGHT:-0.25}" \
  --contrastive-loss-weight "${SUCC_MOTIF_ATTACH_CONTRASTIVE_WEIGHT:-0.20}" \
  --contrastive-margin "${SUCC_MOTIF_ATTACH_CONTRASTIVE_MARGIN:-0.20}" \
  --gate-min-constraint-codes "${SUCC_MOTIF_ATTACH_MIN_CONSTRAINT_CODES:-3}" \
  --gate-min-motif-codes "${SUCC_MOTIF_ATTACH_MIN_MOTIF_CODES:-4}" \
  --source-anchored \
  --connected-region \
  --categorical-delta \
  --valence-budget \
  --motif-attachment \
  --num-attempts 20 \
  --sample-batch-size "${SUCC_MOTIF_ATTACH_SAMPLE_BATCH_SIZE:-5}" \
  --seed "$SEED" \
  --device auto
