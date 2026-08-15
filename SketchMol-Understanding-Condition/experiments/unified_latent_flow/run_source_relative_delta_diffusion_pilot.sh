#!/usr/bin/env bash
# B21: source-relative sparse delta diffusion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_DELTA_DIFFUSION_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_DELTA_DIFFUSION_SEED:-1755}"
OUTPUT_DIR="${SUCC_DELTA_DIFFUSION_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/source_relative_delta_diffusion_v21/seed_${SEED}}"
DATASET_DIR="${SUCC_DELTA_DIFFUSION_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_DELTA_DIFFUSION_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B21 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_DELTA_DIFFUSION_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed B21 run exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/source_relative_delta_diffusion.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_DELTA_DIFFUSION_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_DELTA_DIFFUSION_VALIDATION_LIMIT:-20}" \
  --validation-exclusion-seed "${SUCC_DELTA_DIFFUSION_EXCLUSION_SEED:-1742}" \
  --validation-selection-seed "${SUCC_DELTA_DIFFUSION_VALIDATION_SEED:-2719}" \
  --train-selection-seed "${SUCC_DELTA_DIFFUSION_TRAIN_SEED:-1741}" \
  --property-counts "${SUCC_DELTA_DIFFUSION_PROPERTY_COUNTS:-2,3}" \
  --transport-dim "${SUCC_DELTA_DIFFUSION_TRANSPORT_DIM:-96}" \
  --hidden-dim "${SUCC_DELTA_DIFFUSION_HIDDEN_DIM:-192}" \
  --message-layers "${SUCC_DELTA_DIFFUSION_MESSAGE_LAYERS:-3}" \
  --epochs "${SUCC_DELTA_DIFFUSION_EPOCHS:-8}" \
  --batch-size "${SUCC_DELTA_DIFFUSION_BATCH_SIZE:-4}" \
  --flow-steps "${SUCC_DELTA_DIFFUSION_FLOW_STEPS:-8}" \
  --diffusion-steps "${SUCC_DELTA_DIFFUSION_STEPS:-8}" \
  --birth-capacity "${SUCC_DELTA_DIFFUSION_BIRTH_CAPACITY:-8}" \
  --sample-temperature "${SUCC_DELTA_DIFFUSION_TEMPERATURE:-0.75}" \
  --flow-loss-weight 0.50 \
  --latent-usage-weight 0.20 \
  --latent-usage-margin 0.10 \
  --latent-variance-weight 0.10 \
  --latent-min-std 0.20 \
  --gate-validity 0.95 \
  --gate-mean-unique-valid 10 \
  --gate-strict-any20 0.25 \
  --gate-3p-strict-any20 0.20 \
  --num-attempts 20 \
  --sample-batch-size 5 \
  --seed "$SEED" \
  --device auto
