#!/usr/bin/env bash
# B24: continuous latent over source attachment sites and train-only fragment tokens.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_FRAGMENT_KERNEL_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_FRAGMENT_KERNEL_SEED:-1761}"
OUTPUT_DIR="${SUCC_FRAGMENT_KERNEL_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/seed_${SEED}}"
DATASET_DIR="${SUCC_FRAGMENT_KERNEL_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_FRAGMENT_KERNEL_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
COVERAGE_DIR="${SUCC_FRAGMENT_KERNEL_COVERAGE_DIR:-$SHARED_PROJECT_DIR/outputs/fragment_attachment_coverage_v24/seed_1741}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$COVERAGE_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B24 kernel input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_FRAGMENT_KERNEL_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed B24 kernel result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/latent_fragment_attachment_kernel.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --coverage-summary "$COVERAGE_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_FRAGMENT_KERNEL_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_FRAGMENT_KERNEL_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_FRAGMENT_KERNEL_PROPERTY_COUNTS:-2,3}" \
  --validation-exclusion-seed "${SUCC_FRAGMENT_KERNEL_EXCLUSION_SEED:-1742}" \
  --validation-selection-seed "${SUCC_FRAGMENT_KERNEL_VALIDATION_SEED:-2719}" \
  --train-selection-seed "${SUCC_FRAGMENT_KERNEL_TRAIN_SEED:-1741}" \
  --fingerprint-bits "${SUCC_FRAGMENT_KERNEL_FINGERPRINT_BITS:-256}" \
  --hidden-dim "${SUCC_FRAGMENT_KERNEL_HIDDEN_DIM:-256}" \
  --epochs "${SUCC_FRAGMENT_KERNEL_EPOCHS:-12}" \
  --batch-size "${SUCC_FRAGMENT_KERNEL_BATCH_SIZE:-64}" \
  --flow-steps "${SUCC_FRAGMENT_KERNEL_FLOW_STEPS:-12}" \
  --site-temperature "${SUCC_FRAGMENT_KERNEL_SITE_TEMPERATURE:-0.80}" \
  --min-core-heavy-atoms 5 \
  --max-variable-heavy-atoms 30 \
  --gate-validity 0.90 \
  --gate-strict-any20 0.45 \
  --gate-3p-strict-any20 0.30 \
  --gate-mean-unique-valid 8 \
  --num-attempts 20 \
  --seed "$SEED" \
  --device auto
