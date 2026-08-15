#!/usr/bin/env bash
# B34: train-only continuous Pareto latent transport and internal-dev signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_CONTINUOUS_PARETO_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="${SUCC_CONTINUOUS_PARETO_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/continuous_pareto_latent_transport_v34/seed_1961}"
DATASET_DIR="${SUCC_CONTINUOUS_PARETO_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"
REPRESENTATION_DIR="${SUCC_CONTINUOUS_PARETO_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_CONTINUOUS_PARETO_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
B31_DIR="${SUCC_CONTINUOUS_PARETO_B31_DIR:-$SHARED_PROJECT_DIR/outputs/assay_joint_site_token_latent_v31/seed_1931}"
B32_DIR="${SUCC_CONTINUOUS_PARETO_B32_DIR:-$SHARED_PROJECT_DIR/outputs/structure_constrained_joint_latent_v32/seed_1941}"
ORACLE_DIR="${SUCC_ASSAY_ORACLE_DIR:-$SHARED_PROJECT_DIR/inputs/tdc_oracles}"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
DRD2_ORACLE="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
PREREGISTRATION="$SCRIPT_DIR/continuous_pareto_latent_transport_v34_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$B31_DIR/assay_joint_site_token_energy.pt" \
  "$B31_DIR/summary.json" \
  "$B32_DIR/structure_feasibility_energy.pt" \
  "$B32_DIR/summary.json" \
  "$GSK3B_ORACLE" \
  "$DRD2_ORACLE" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B34 input: $path" >&2; exit 2; }
done

[[ ! -f "$OUTPUT_DIR/summary.json" ]] || {
  echo "ERROR: completed B34 result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/continuous_pareto_latent_transport.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --b31-checkpoint "$B31_DIR/assay_joint_site_token_energy.pt" \
  --b31-summary "$B31_DIR/summary.json" \
  --b32-checkpoint "$B32_DIR/structure_feasibility_energy.pt" \
  --b32-summary "$B32_DIR/summary.json" \
  --gsk3b-oracle "$GSK3B_ORACLE" \
  --drd2-oracle "$DRD2_ORACLE" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu
