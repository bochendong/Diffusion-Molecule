#!/usr/bin/env bash
# Execute the final target-free V11 representation gate or its science gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_V11_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_V11_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_V11_STAGE:?Set SUCC_V11_STAGE to execute or gate}"
OUTPUT_ROOT="${SUCC_V11_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/elementwise_additive_semantic_set_compiler_v11/seed_2161}"
EXECUTION_DIR="$OUTPUT_ROOT/execution"
GATE_DIR="$OUTPUT_ROOT/gate"
SEMANTIC_ROOT="$SHARED_PROJECT_DIR/outputs/semantic_energy_graph_jump_v1/seed_2045"
V5_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_property_set_router_v5/seed_2071"
V6_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_router_table1_bridge_v6/seed_2081"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
SFT_ADAPTER_DIR="$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter"
PROTOCOL="$SCRIPT_DIR/elementwise_additive_semantic_set_compiler_v11_preregistration.json"
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

case "$STAGE" in
  execute)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/elementwise_additive_semantic_set_compiler_v11.py" \
      --protocol-manifest "$PROTOCOL" \
      --basis-bundle "$V6_ROOT/prepare/target_free_generation_bundle.pt" \
      --source-manifest "$SEMANTIC_ROOT/prepare/generation_conditions.json" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
      --sft-adapter-dir "$SFT_ADAPTER_DIR" \
      --v5-root "$V5_ROOT" \
      --e1-manifest "$E1_MANIFEST" \
      --output-dir "$EXECUTION_DIR" \
      --device auto
    ;;
  gate)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/gate_elementwise_additive_semantic_set_compiler_v11.py" \
      --protocol-manifest "$PROTOCOL" \
      --execution-summary "$EXECUTION_DIR/summary.json" \
      --output-dir "$GATE_DIR"
    ;;
  *)
    echo "ERROR: unsupported V11 stage: $STAGE" >&2
    exit 2
    ;;
esac
