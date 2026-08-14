#!/usr/bin/env bash
# Grammar-native valence-budget hierarchical VQ pilot, exact n=20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_VALENCE_HIER_VQ_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_VALENCE_HIER_VQ_SEED:-1741}"
OUTPUT_DIR="${SUCC_VALENCE_HIER_VQ_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/valence_budget_hierarchical_vq_graph_flow_v10/seed_${SEED}}"
DATASET_DIR="${SUCC_VALENCE_HIER_VQ_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_VALENCE_HIER_VQ_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing valence-budget VQ input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_VALENCE_HIER_VQ_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed valence-budget VQ run exists: $OUTPUT_DIR/summary.json" >&2
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
  --train-limit "${SUCC_VALENCE_HIER_VQ_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_VALENCE_HIER_VQ_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_VALENCE_HIER_VQ_PROPERTY_COUNTS:-2,3}" \
  --constraint-codebook-size "${SUCC_VALENCE_HIER_VQ_CONSTRAINT_CODEBOOK_SIZE:-16}" \
  --constraint-code-dim "${SUCC_VALENCE_HIER_VQ_CONSTRAINT_CODE_DIM:-32}" \
  --motif-codebook-size "${SUCC_VALENCE_HIER_VQ_MOTIF_CODEBOOK_SIZE:-64}" \
  --motif-code-dim "${SUCC_VALENCE_HIER_VQ_MOTIF_CODE_DIM:-64}" \
  --epochs "${SUCC_VALENCE_HIER_VQ_EPOCHS:-8}" \
  --batch-size "${SUCC_VALENCE_HIER_VQ_BATCH_SIZE:-6}" \
  --hidden-dim "${SUCC_VALENCE_HIER_VQ_HIDDEN_DIM:-256}" \
  --sampling-temperature "${SUCC_VALENCE_HIER_VQ_TEMPERATURE:-0.80}" \
  --edit-gate-loss-weight "${SUCC_VALENCE_HIER_VQ_REGION_WEIGHT:-0.50}" \
  --delta-loss-weight "${SUCC_VALENCE_HIER_VQ_DELTA_WEIGHT:-0.50}" \
  --valence-budget-loss-weight "${SUCC_VALENCE_HIER_VQ_BUDGET_WEIGHT:-0.25}" \
  --contrastive-loss-weight "${SUCC_VALENCE_HIER_VQ_CONTRASTIVE_WEIGHT:-0.20}" \
  --contrastive-margin "${SUCC_VALENCE_HIER_VQ_CONTRASTIVE_MARGIN:-0.20}" \
  --gate-min-constraint-codes "${SUCC_VALENCE_HIER_VQ_MIN_CONSTRAINT_CODES:-3}" \
  --gate-min-motif-codes "${SUCC_VALENCE_HIER_VQ_MIN_MOTIF_CODES:-4}" \
  --source-anchored \
  --connected-region \
  --categorical-delta \
  --valence-budget \
  --num-attempts 20 \
  --sample-batch-size "${SUCC_VALENCE_HIER_VQ_SAMPLE_BATCH_SIZE:-5}" \
  --seed "$SEED" \
  --device auto
