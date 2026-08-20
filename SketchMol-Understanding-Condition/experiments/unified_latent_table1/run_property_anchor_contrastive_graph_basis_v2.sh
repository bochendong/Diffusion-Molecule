#!/usr/bin/env bash
# Run one bounded anchored/contrastive property-coordinate representation probe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_PROPERTY_ANCHOR_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_PROPERTY_ANCHOR_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
PREVIOUS_ROOT="$SHARED_PROJECT_DIR/outputs/semantic_energy_graph_jump_v1/seed_2045"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
SFT_ADAPTER_DIR="${SUCC_PROPERTY_ANCHOR_SFT_ADAPTER_DIR:-$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter}"
OUTPUT_DIR="${SUCC_PROPERTY_ANCHOR_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/property_anchor_contrastive_graph_basis_v2/seed_2053}"
PROTOCOL="$SCRIPT_DIR/property_anchor_contrastive_graph_basis_v2_preregistration.json"
E1_MANIFEST="$SCRIPT_DIR/e1_nl_condition_head_preregistration.json"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

export PYTHONPATH="$DEP_OVERLAY:$PROJECT_DIR:$PROJECT_DIR/experiments/unified_latent_flow:$SCRIPT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/property_anchor_contrastive_graph_basis_v2.py" \
  --protocol-manifest "$PROTOCOL" \
  --prepare-summary "$PREVIOUS_ROOT/prepare/summary.json" \
  --fit-probe-bundle "$PREVIOUS_ROOT/prepare/fit_probe_bundle.pt" \
  --embedding-cache "$PREVIOUS_ROOT/train/constraint_embeddings.pt" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
  --sft-adapter-dir "$SFT_ADAPTER_DIR" \
  --e1-manifest "$E1_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --device auto
