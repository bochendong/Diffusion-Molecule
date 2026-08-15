#!/usr/bin/env bash
# B23: evaluate a latent-conditioned local rewrite support on the frozen B22 model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_EDIT_CENTER_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_EDIT_CENTER_SEED:-1759}"
OUTPUT_DIR="${SUCC_EDIT_CENTER_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_edit_center_rewrite_decode_v23/seed_${SEED}}"
DATASET_DIR="${SUCC_EDIT_CENTER_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_EDIT_CENTER_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
B22_DIR="${SUCC_EDIT_CENTER_B22_DIR:-$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757}"

for path in \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B23 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_EDIT_CENTER_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed B23 run exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/latent_edit_center_rewrite_decode.py" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --delta-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  --delta-summary "$B22_DIR/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --validation-limit "${SUCC_EDIT_CENTER_VALIDATION_LIMIT:-20}" \
  --validation-exclusion-seed "${SUCC_EDIT_CENTER_EXCLUSION_SEED:-1742}" \
  --validation-selection-seed "${SUCC_EDIT_CENTER_VALIDATION_SEED:-2719}" \
  --property-counts "${SUCC_EDIT_CENTER_PROPERTY_COUNTS:-2,3}" \
  --flow-steps "${SUCC_EDIT_CENTER_FLOW_STEPS:-8}" \
  --diffusion-steps "${SUCC_EDIT_CENTER_DIFFUSION_STEPS:-8}" \
  --birth-capacity "${SUCC_EDIT_CENTER_BIRTH_CAPACITY:-8}" \
  --sample-temperature "${SUCC_EDIT_CENTER_TEMPERATURE:-0.75}" \
  --rewrite-radius "${SUCC_EDIT_CENTER_RADIUS:-1}" \
  --centers-per-extra-property "${SUCC_EDIT_CENTER_PER_EXTRA_PROPERTY:-1}" \
  --gate-validity-improvement 0.10 \
  --gate-strict-retention -0.05 \
  --gate-3p-strict-any20 0.14 \
  --num-attempts 20 \
  --sample-batch-size 5 \
  --seed "$SEED" \
  --device auto
