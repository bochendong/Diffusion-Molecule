#!/usr/bin/env bash
# B31: train-only assay supervision for a direct joint site-token latent draw.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
PYTHON_BIN="${SUCC_ASSAY_JOINT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="${SUCC_ASSAY_JOINT_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/assay_joint_site_token_latent_v31/seed_1931}"
DATASET_DIR="${SUCC_ASSAY_JOINT_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
TABLE1_EVAL="${SUCC_ASSAY_JOINT_TABLE1_EVAL:-$DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
REPRESENTATION_DIR="${SUCC_ASSAY_JOINT_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_ASSAY_JOINT_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
B29_DIR="${SUCC_ASSAY_JOINT_B29_DIR:-$SHARED_PROJECT_DIR/outputs/table1_energy_tilted_latent_transfer_v29/seed_1911}"
B30_DIR="${SUCC_ASSAY_JOINT_B30_DIR:-$SHARED_PROJECT_DIR/outputs/table1_assay_latent_action_support_v30_r1/seed_1922/merged}"
ORACLE_DIR="${SUCC_ASSAY_ORACLE_DIR:-$SHARED_PROJECT_DIR/inputs/tdc_oracles}"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
DRD2_ORACLE="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
PREREGISTRATION="$SCRIPT_DIR/assay_joint_site_token_latent_v31_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$TABLE1_EVAL" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$B29_DIR/summary.json" \
  "$B30_DIR/summary.json" \
  "$GSK3B_ORACLE" \
  "$DRD2_ORACLE" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B31 input: $path" >&2; exit 2; }
done

[[ ! -f "$OUTPUT_DIR/summary.json" ]] || {
  echo "ERROR: completed B31 result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/assay_joint_site_token_latent.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --table1-eval-csv "$TABLE1_EVAL" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --b29-summary "$B29_DIR/summary.json" \
  --b30-summary "$B30_DIR/summary.json" \
  --gsk3b-oracle "$GSK3B_ORACLE" \
  --drd2-oracle "$DRD2_ORACLE" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu
