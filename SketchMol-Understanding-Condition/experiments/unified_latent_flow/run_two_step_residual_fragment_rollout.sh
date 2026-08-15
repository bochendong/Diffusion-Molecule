#!/usr/bin/env bash
# B25: target-blind second fragment action for 3-property requests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_TWO_STEP_FRAGMENT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_TWO_STEP_FRAGMENT_SEED:-1761}"
OUTPUT_DIR="${SUCC_TWO_STEP_FRAGMENT_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/two_step_residual_fragment_rollout_v25/seed_${SEED}}"
DATASET_DIR="${SUCC_TWO_STEP_FRAGMENT_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_TWO_STEP_FRAGMENT_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
B24_DIR="${SUCC_TWO_STEP_FRAGMENT_B24_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B24_DIR/latent_fragment_attachment_kernel.pt" \
  "$B24_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B25 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_TWO_STEP_FRAGMENT_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed B25 result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/two_step_residual_fragment_rollout.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$B24_DIR/latent_fragment_attachment_kernel.pt" \
  --fragment-summary "$B24_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --validation-limit "${SUCC_TWO_STEP_FRAGMENT_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_TWO_STEP_FRAGMENT_PROPERTY_COUNTS:-2,3}" \
  --validation-exclusion-seed "${SUCC_TWO_STEP_FRAGMENT_EXCLUSION_SEED:-1742}" \
  --validation-selection-seed "${SUCC_TWO_STEP_FRAGMENT_VALIDATION_SEED:-2719}" \
  --fingerprint-bits 256 \
  --flow-steps 12 \
  --site-temperature 0.80 \
  --min-core-heavy-atoms 5 \
  --max-variable-heavy-atoms 30 \
  --gate-validity 0.95 \
  --gate-overall-strict-delta -0.05 \
  --gate-3p-strict-delta 0.14 \
  --gate-mean-unique-valid 15 \
  --num-attempts 20 \
  --seed "$SEED" \
  --device auto
