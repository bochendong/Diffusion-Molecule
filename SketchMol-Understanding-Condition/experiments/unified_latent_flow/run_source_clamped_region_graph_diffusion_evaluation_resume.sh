#!/usr/bin/env bash
# Resume B37 from the immutable 4,700-row frozen candidate CSV; CPU only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_REGION_RESUME_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="$SHARED_PROJECT_DIR/outputs/source_clamped_region_graph_diffusion_v37/seed_1983"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B36_DIR="$SHARED_PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
FAILED_LOG="$SHARED_PROJECT_DIR/logs/source_clamped_region_graph_diffusion_v37/uca-region-diff-v37-19864238.log"
FROZEN="$OUTPUT_DIR/frozen_train_only_dev_candidates.csv"
ORIGINAL_PREREG="$SCRIPT_DIR/source_clamped_region_graph_diffusion_v37_preregistration.json"
RESUME_MANIFEST="$SCRIPT_DIR/source_clamped_region_graph_diffusion_v37r1_resume_manifest.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json" \
  "$B36_DIR/summary.json" \
  "$FAILED_LOG" \
  "$FROZEN" \
  "$ORIGINAL_PREREG" \
  "$RESUME_MANIFEST"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B37r1 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
  echo "ERROR: completed B37 summary exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/resume_source_clamped_region_graph_diffusion_evaluation.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  --b22-summary "$B22_DIR/summary.json" \
  --b36-summary "$B36_DIR/summary.json" \
  --original-preregistration "$ORIGINAL_PREREG" \
  --failed-log "$FAILED_LOG" \
  --frozen-candidates "$FROZEN" \
  --resume-manifest "$RESUME_MANIFEST" \
  --output-dir "$OUTPUT_DIR"
