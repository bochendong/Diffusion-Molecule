#!/usr/bin/env bash
# Run one physically separated V6 bridge stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_V6_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_V6_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_V6_STAGE:?Set SUCC_V6_STAGE}"
OUTPUT_ROOT="${SUCC_V6_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/mass_conserving_router_table1_bridge_v6/seed_2081}"
PREPARE_DIR="$OUTPUT_ROOT/prepare"
FROZEN_DIR="$OUTPUT_ROOT/frozen"
EVALUATION_DIR="$OUTPUT_ROOT/evaluation"
GATE_DIR="$OUTPUT_ROOT/gate"
SEMANTIC_ROOT="$SHARED_PROJECT_DIR/outputs/semantic_energy_graph_jump_v1/seed_2045"
V5_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_property_set_router_v5/seed_2071"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
SFT_ADAPTER_DIR="$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter"
PROTOCOL="$SCRIPT_DIR/mass_conserving_router_table1_bridge_v6_preregistration.json"

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

COMMON=("$PYTHON_BIN" "$SCRIPT_DIR/mass_conserving_router_table1_bridge_v6.py" --protocol-manifest "$PROTOCOL")

case "$STAGE" in
  prepare)
    exec "${COMMON[@]}" prepare \
      --fit-probe-bundle "$SEMANTIC_ROOT/prepare/fit_probe_bundle.pt" \
      --generation-conditions "$SEMANTIC_ROOT/prepare/generation_conditions.json" \
      --v5-summary "$V5_ROOT/full/summary.json" \
      --v5-gate "$V5_ROOT/gate/gate_summary.json" \
      --v5-unlock "$V5_ROOT/gate/generation_unlock.json" \
      --output-dir "$PREPARE_DIR"
    ;;
  freeze)
    exec "${COMMON[@]}" freeze \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --generation-bundle "$PREPARE_DIR/target_free_generation_bundle.pt" \
      --generation-conditions "$SEMANTIC_ROOT/prepare/generation_conditions.json" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
      --sft-adapter-dir "$SFT_ADAPTER_DIR" \
      --v5-lora-adapter-dir "$V5_ROOT/full/lora_adapter" \
      --v5-router-checkpoint "$V5_ROOT/full/structured_sparse_router.pt" \
      --v5-summary "$V5_ROOT/full/summary.json" \
      --v5-gate "$V5_ROOT/gate/gate_summary.json" \
      --v5-unlock "$V5_ROOT/gate/generation_unlock.json" \
      --output-dir "$FROZEN_DIR" \
      --device auto
    ;;
  evaluate)
    exec "${COMMON[@]}" evaluate \
      --evaluation-targets "$SEMANTIC_ROOT/prepare/sealed_evaluation_targets.json" \
      --frozen-root "$FROZEN_DIR" \
      --output-dir "$EVALUATION_DIR"
    ;;
  gate)
    exec "${COMMON[@]}" gate \
      --evaluation-summary "$EVALUATION_DIR/summary.json" \
      --output-dir "$GATE_DIR"
    ;;
  *)
    echo "ERROR: unsupported V6 stage: $STAGE" >&2
    exit 2
    ;;
esac
