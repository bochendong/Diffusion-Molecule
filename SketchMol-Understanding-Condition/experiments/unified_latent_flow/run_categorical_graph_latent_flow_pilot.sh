#!/usr/bin/env bash
# Fast source-conditioned graph-latent rectified-flow signal on held-out edits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_GRAPH_FLOW_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_GRAPH_FLOW_SEED:-1727}"
OUTPUT_DIR="${SUCC_GRAPH_FLOW_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/categorical_graph_latent_flow_v1/seed_${SEED}}"
DATASET_DIR="${SUCC_GRAPH_FLOW_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_GRAPH_FLOW_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing graph-flow input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_GRAPH_FLOW_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed graph-flow pilot exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/categorical_graph_latent_flow.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_GRAPH_FLOW_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_GRAPH_FLOW_VALIDATION_LIMIT:-16}" \
  --property-counts "${SUCC_GRAPH_FLOW_PROPERTY_COUNTS:-2,3}" \
  --epochs "${SUCC_GRAPH_FLOW_EPOCHS:-2}" \
  --batch-size "${SUCC_GRAPH_FLOW_BATCH_SIZE:-8}" \
  --hidden-dim "${SUCC_GRAPH_FLOW_HIDDEN_DIM:-256}" \
  --flow-steps "${SUCC_GRAPH_FLOW_STEPS:-6}" \
  --num-attempts 20 \
  --sample-batch-size "${SUCC_GRAPH_FLOW_SAMPLE_BATCH_SIZE:-5}" \
  --seed "$SEED" \
  --device auto
