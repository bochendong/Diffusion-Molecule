#!/usr/bin/env bash
# Train/evaluate one CPU-only exact-20 property-aligned balanced transport.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_BALANCED_TXN_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SEED="${SUCC_BALANCED_TXN_SEED:-2011}"
OUTPUT_DIR="${SUCC_BALANCED_TXN_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/property_aligned_balanced_transaction_transport_v1/seed_${SEED}}"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B36_DIR="$SHARED_PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
B41_DIR="$SHARED_PROJECT_DIR/outputs/viability_preserving_interacting_particle_transport_v41/seed_1991"
EVIDENCE_DIR="$SHARED_PROJECT_DIR/outputs/set_closed_graph_rewrite_evidence_v1/seed_2001"
B43_DIR="$SHARED_PROJECT_DIR/outputs/set_closed_graph_transport_v1/seed_2003"
ATOMIC_DIR="$SHARED_PROJECT_DIR/outputs/atomic_closed_transaction_latent_decoder_v1/seed_2005"
VQ_DIR="$SHARED_PROJECT_DIR/outputs/compositional_closed_transaction_vq_flow_v1/seed_2007"
PROBE_DIR="$SHARED_PROJECT_DIR/outputs/compositional_closed_reaction_template_probe_v1"
PREREGISTRATION="$SCRIPT_DIR/property_aligned_balanced_transaction_transport_v1_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json" \
  "$B36_DIR/train_patch_records.jsonl" \
  "$B41_DIR/viability_interacting_particle_transport.pt" \
  "$B41_DIR/summary.json" \
  "$EVIDENCE_DIR/summary.json" \
  "$EVIDENCE_DIR/rewrite_records.jsonl" \
  "$B43_DIR/set_closed_graph_transport.pt" \
  "$B43_DIR/summary.json" \
  "$ATOMIC_DIR/atomic_closed_transaction_decoder.pt" \
  "$ATOMIC_DIR/summary.json" \
  "$PROBE_DIR/source_disjoint_support_48.json" \
  "$PROBE_DIR/source_disjoint_support_radius0_48.json" \
  "$VQ_DIR/summary.json" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing balanced-transport input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_BALANCED_TXN_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed balanced-transport result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/property_aligned_balanced_transaction_transport.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  --b22-summary "$B22_DIR/summary.json" \
  --b36-records "$B36_DIR/train_patch_records.jsonl" \
  --b41-checkpoint "$B41_DIR/viability_interacting_particle_transport.pt" \
  --b41-summary "$B41_DIR/summary.json" \
  --set-evidence-summary "$EVIDENCE_DIR/summary.json" \
  --set-evidence-records "$EVIDENCE_DIR/rewrite_records.jsonl" \
  --b43-checkpoint "$B43_DIR/set_closed_graph_transport.pt" \
  --b43-summary "$B43_DIR/summary.json" \
  --atomic-checkpoint "$ATOMIC_DIR/atomic_closed_transaction_decoder.pt" \
  --atomic-summary "$ATOMIC_DIR/summary.json" \
  --radius-one-support-probe "$PROBE_DIR/source_disjoint_support_48.json" \
  --radius-zero-support-probe "$PROBE_DIR/source_disjoint_support_radius0_48.json" \
  --vq-summary "$VQ_DIR/summary.json" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu
