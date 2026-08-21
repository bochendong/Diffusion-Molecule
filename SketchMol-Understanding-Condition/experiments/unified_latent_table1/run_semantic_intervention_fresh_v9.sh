#!/usr/bin/env bash
# Run one physically isolated V9 stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LATENT_DIR="$PROJECT_DIR/experiments/unified_latent_flow"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_V9_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_V9_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_V9_STAGE:?Set SUCC_V9_STAGE}"
OUTPUT_ROOT="${SUCC_V9_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/semantic_intervention_fresh_v9/seed_2121}"
PREPARE_DIR="$OUTPUT_ROOT/prepare"
REPLICATE_INDEX="${SLURM_ARRAY_TASK_ID:-${SUCC_V9_REPLICATE_INDEX:-0}}"
REPLICATE_ROOT="$OUTPUT_ROOT/replicate_$REPLICATE_INDEX"
FROZEN_DIR="$REPLICATE_ROOT/frozen"
EVALUATION_DIR="$REPLICATE_ROOT/evaluation"
GATE_DIR="$OUTPUT_ROOT/gate"

DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
SFT_ADAPTER_DIR="$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter"
V5_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_property_set_router_v5/seed_2071"
V6_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_router_table1_bridge_v6/seed_2081"
V7_ROOT="$SHARED_PROJECT_DIR/outputs/mass_conserving_router_fresh_horizon_v7/seed_2093"
V3_ROOT="$SHARED_PROJECT_DIR/outputs/language_grounded_graph_latent_fresh_edit_v3/seed_2022"
B36_ROOT="$SHARED_PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
B26_ROOT="$SHARED_PROJECT_DIR/outputs/frozen_fragment_attachment_fresh_holdout_v26/seed_1873"
B33_ROOT="$SHARED_PROJECT_DIR/outputs/pareto_conditioned_joint_latent_v33/seed_1951"
FRESH_V1_ROOT="$SHARED_PROJECT_DIR/outputs/fresh_graph_jump_language_confirmation_v1/seed_2032"
SEMANTIC_ROOT="$SHARED_PROJECT_DIR/outputs/semantic_energy_graph_jump_v1/seed_2045"
D0_ROOT="$SHARED_PROJECT_DIR/outputs/d0_b41_table1_n20"
E1_MANIFEST="$SCRIPT_DIR/e1_nl_condition_head_preregistration.json"
PROTOCOL="$SCRIPT_DIR/semantic_intervention_fresh_v9_preregistration.json"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

export PYTHONPATH="$DEP_OVERLAY:$PROJECT_DIR:$LATENT_DIR:$SCRIPT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

COMMON=("$PYTHON_BIN" "$SCRIPT_DIR/semantic_intervention_fresh_v9.py" --protocol-manifest "$PROTOCOL")

case "$STAGE" in
  prepare)
    exec "${COMMON[@]}" prepare \
      --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
      --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
      --b36-records "$B36_ROOT/train_patch_records.jsonl" \
      --predecessor-fit-bundle "$V3_ROOT/prepare/fit_only_direction_pairs.pt" \
      --e1-manifest "$E1_MANIFEST" \
      --v6-basis-bundle "$V6_ROOT/prepare/target_free_generation_bundle.pt" \
      --known-source "$B26_ROOT/validation_candidates.csv" \
      --known-source "$B33_ROOT/fresh_internal_b33_pareto_candidates.csv" \
      --known-source "$V3_ROOT/prepare/generation_conditions.json" \
      --known-source "$FRESH_V1_ROOT/prepare/generation_conditions.json" \
      --known-source "$SEMANTIC_ROOT/prepare/generation_conditions.json" \
      --known-source "$D0_ROOT/d0_b41_table1_n20_candidates.csv" \
      --known-source "$V7_ROOT/prepare/generation_conditions.json" \
      --output-dir "$PREPARE_DIR"
    ;;
  freeze)
    exec "${COMMON[@]}" freeze \
      --replicate-index "$REPLICATE_INDEX" \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --generation-conditions "$PREPARE_DIR/generation_conditions.json" \
      --v6-basis-bundle "$V6_ROOT/prepare/target_free_generation_bundle.pt" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
      --sft-adapter-dir "$SFT_ADAPTER_DIR" \
      --v5-root "$V5_ROOT" \
      --v5-gate "$V5_ROOT/gate/gate_summary.json" \
      --v5-unlock "$V5_ROOT/gate/generation_unlock.json" \
      --output-dir "$FROZEN_DIR" \
      --device auto
    ;;
  evaluate)
    exec "${COMMON[@]}" evaluate \
      --replicate-index "$REPLICATE_INDEX" \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --evaluation-targets "$PREPARE_DIR/sealed_evaluation_targets.json" \
      --frozen-root "$FROZEN_DIR" \
      --output-dir "$EVALUATION_DIR"
    ;;
  gate)
    exec "${COMMON[@]}" gate \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --evaluation-root "$OUTPUT_ROOT" \
      --output-dir "$GATE_DIR"
    ;;
  *)
    echo "ERROR: unsupported V9 stage: $STAGE" >&2
    exit 2
    ;;
esac
