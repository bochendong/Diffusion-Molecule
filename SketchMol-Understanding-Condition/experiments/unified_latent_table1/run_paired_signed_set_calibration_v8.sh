#!/usr/bin/env bash
# Run one physically isolated V8 calibration stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_V8_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_V8_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_V8_STAGE:?Set SUCC_V8_STAGE}"
REPLICATE_INDEX="${SLURM_ARRAY_TASK_ID:-${SUCC_V8_REPLICATE_INDEX:-0}}"
OUTPUT_ROOT="${SUCC_V8_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/paired_signed_set_calibration_v8/seed_2111}"
REPLICATE_ROOT="$OUTPUT_ROOT/replicate_$REPLICATE_INDEX"

SEMANTIC_ROOT="$SHARED_PROJECT_DIR/outputs/semantic_energy_graph_jump_v1/seed_2045"
V5_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_property_set_router_v5/seed_2071"
V6_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_router_table1_bridge_v6/seed_2081"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
SFT_ADAPTER_DIR="$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter"
PROTOCOL="$SCRIPT_DIR/paired_signed_set_calibration_v8_preregistration.json"

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

COMMON=("$PYTHON_BIN" "$SCRIPT_DIR/paired_signed_set_calibration_v8.py" --protocol-manifest "$PROTOCOL")
SHARED_INPUTS=(
  --v6-prepare-summary "$V6_ROOT/prepare/summary.json"
  --v6-basis-bundle "$V6_ROOT/prepare/target_free_generation_bundle.pt"
  --generation-conditions "$SEMANTIC_ROOT/prepare/generation_conditions.json"
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt"
  --representation-summary "$REPRESENTATION_DIR/summary.json"
  --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt"
  --sft-adapter-dir "$SFT_ADAPTER_DIR"
  --v5-lora-adapter-dir "$V5_ROOT/full/lora_adapter"
  --v5-router-checkpoint "$V5_ROOT/full/structured_sparse_router.pt"
  --v5-summary "$V5_ROOT/full/summary.json"
  --v5-gate "$V5_ROOT/gate/gate_summary.json"
  --v5-unlock "$V5_ROOT/gate/generation_unlock.json"
)

case "$STAGE" in
  validate)
    exec "${COMMON[@]}" validate \
      "${SHARED_INPUTS[@]}" \
      --evaluation-targets "$SEMANTIC_ROOT/prepare/sealed_evaluation_targets.json" \
      --output-root "$OUTPUT_ROOT"
    ;;
  freeze)
    exec "${COMMON[@]}" freeze \
      --replicate-index "$REPLICATE_INDEX" \
      "${SHARED_INPUTS[@]}" \
      --output-dir "$REPLICATE_ROOT/frozen" \
      --device auto
    ;;
  evaluate)
    exec "${COMMON[@]}" evaluate \
      --replicate-index "$REPLICATE_INDEX" \
      --evaluation-targets "$SEMANTIC_ROOT/prepare/sealed_evaluation_targets.json" \
      --frozen-root "$REPLICATE_ROOT/frozen" \
      --output-dir "$REPLICATE_ROOT/evaluation"
    ;;
  gate)
    exec "${COMMON[@]}" gate \
      --evaluation-root "$OUTPUT_ROOT" \
      --output-dir "$OUTPUT_ROOT/gate"
    ;;
  *)
    echo "ERROR: unsupported V8 stage: $STAGE" >&2
    exit 2
    ;;
esac
