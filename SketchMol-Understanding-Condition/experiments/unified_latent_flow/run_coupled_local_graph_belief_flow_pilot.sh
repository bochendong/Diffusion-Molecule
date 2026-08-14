#!/usr/bin/env bash
# Coupled node/incident-edge categorical graph-belief pilot, 2p only, exact n=20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_COUPLED_BELIEF_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_COUPLED_BELIEF_SEED:-1733}"
OUTPUT_DIR="${SUCC_COUPLED_BELIEF_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/coupled_local_graph_belief_flow_v4/seed_${SEED}}"
DATASET_DIR="${SUCC_COUPLED_BELIEF_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_COUPLED_BELIEF_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing coupled-belief input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_COUPLED_BELIEF_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed coupled-belief pilot exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/coupled_local_graph_belief_flow.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_COUPLED_BELIEF_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_COUPLED_BELIEF_VALIDATION_LIMIT:-12}" \
  --property-counts "${SUCC_COUPLED_BELIEF_PROPERTY_COUNTS:-2}" \
  --epochs "${SUCC_COUPLED_BELIEF_EPOCHS:-8}" \
  --batch-size "${SUCC_COUPLED_BELIEF_BATCH_SIZE:-8}" \
  --hidden-dim "${SUCC_COUPLED_BELIEF_HIDDEN_DIM:-256}" \
  --sampling-temperature "${SUCC_COUPLED_BELIEF_TEMPERATURE:-0.70}" \
  --num-attempts 20 \
  --sample-batch-size "${SUCC_COUPLED_BELIEF_SAMPLE_BATCH_SIZE:-5}" \
  --seed "$SEED" \
  --device auto
