#!/usr/bin/env bash
# B29: zero-training target-free Table1 transfer of frozen B24/B27/B28.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
PYTHON_BIN="${SUCC_TABLE1_LATENT_TRANSFER_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="${SUCC_TABLE1_LATENT_TRANSFER_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/table1_energy_tilted_latent_transfer_v29/seed_1911}"
DATASET_DIR="${SUCC_TABLE1_LATENT_TRANSFER_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
TABLE1_EVAL="${SUCC_TABLE1_LATENT_TRANSFER_EVAL:-$DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
REPRESENTATION_DIR="${SUCC_TABLE1_LATENT_TRANSFER_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_TABLE1_LATENT_TRANSFER_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
ENERGY_DIR="${SUCC_TABLE1_LATENT_TRANSFER_ENERGY_DIR:-$SHARED_PROJECT_DIR/outputs/latent_property_energy_guidance_v27/seed_1891}"
PREREGISTRATION="$SCRIPT_DIR/table1_energy_tilted_latent_transfer_v29_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$TABLE1_EVAL" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$ENERGY_DIR/latent_property_energy.pt" \
  "$ENERGY_DIR/summary.json" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B29 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
  echo "ERROR: completed B29 result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/table1_energy_tilted_latent_transfer.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --table1-eval-csv "$TABLE1_EVAL" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --energy-checkpoint "$ENERGY_DIR/latent_property_energy.pt" \
  --energy-summary "$ENERGY_DIR/summary.json" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu
