#!/usr/bin/env bash
# B28: one sampled token from distance-prior times train-only energy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_ENERGY_TILTED_VQ_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="${SUCC_ENERGY_TILTED_VQ_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/energy_tilted_vq_fragment_sampling_v28/seed_1901}"
DATASET_DIR="${SUCC_ENERGY_TILTED_VQ_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_ENERGY_TILTED_VQ_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_ENERGY_TILTED_VQ_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
ENERGY_DIR="${SUCC_ENERGY_TILTED_VQ_ENERGY_DIR:-$SHARED_PROJECT_DIR/outputs/latent_property_energy_guidance_v27/seed_1891}"
PREREGISTRATION="$SCRIPT_DIR/energy_tilted_vq_v28_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$ENERGY_DIR/latent_property_energy.pt" \
  "$ENERGY_DIR/summary.json" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B28 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
  echo "ERROR: completed B28 result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/energy_tilted_vq_fragment_sampling.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --energy-checkpoint "$ENERGY_DIR/latent_property_energy.pt" \
  --energy-summary "$ENERGY_DIR/summary.json" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu
