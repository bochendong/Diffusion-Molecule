#!/usr/bin/env bash
# Run one physically separated stage of the fresh graph-jump/language confirmation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LATENT_DIR="$PROJECT_DIR/experiments/unified_latent_flow"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_FRESH_CONFIRM_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_FRESH_CONFIRM_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_FRESH_CONFIRM_STAGE:?Set SUCC_FRESH_CONFIRM_STAGE}"
ARM_GROUP="${SUCC_FRESH_CONFIRM_ARM_GROUP:-}"
OUTPUT_ROOT="${SUCC_FRESH_CONFIRM_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/fresh_graph_jump_language_confirmation_v1/seed_2032}"
PREPARE_DIR="$OUTPUT_ROOT/prepare"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B36_DIR="$SHARED_PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
STATE_GUIDANCE_DIR="$SHARED_PROJECT_DIR/outputs/common_llm_state_viability_guidance_v1/seed_1995/prepare"
B41_DIR="$SHARED_PROJECT_DIR/outputs/viability_preserving_interacting_particle_transport_v41/seed_1991"
B26_DIR="$SHARED_PROJECT_DIR/outputs/frozen_fragment_attachment_fresh_holdout_v26/seed_1873"
B33_DIR="$SHARED_PROJECT_DIR/outputs/pareto_conditioned_joint_latent_v33/seed_1951"
V3_DIR="$SHARED_PROJECT_DIR/outputs/language_grounded_graph_latent_fresh_edit_v3/seed_2022"
D0_DIR="$SHARED_PROJECT_DIR/outputs/d0_b41_table1_n20"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
D3_DIR="$SHARED_PROJECT_DIR/outputs/d3_event_kernel_energy_grpo_table1_n20"
E1_DIR="$SHARED_PROJECT_DIR/outputs/e1_nl_condition_head_table1_n20"
PREREGISTRATION="$SCRIPT_DIR/fresh_graph_jump_language_confirmation_v1_preregistration.json"
PREDECESSOR="$LATENT_DIR/language_grounded_graph_latent_flow_v1_preregistration.json"
E1_MANIFEST="$SCRIPT_DIR/e1_nl_condition_head_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json" \
  "$B36_DIR/train_patch_records.jsonl" \
  "$STATE_GUIDANCE_DIR/fit_only_trajectories.pt" \
  "$B41_DIR/viability_interacting_particle_transport.pt" \
  "$B26_DIR/validation_candidates.csv" \
  "$B33_DIR/fresh_internal_b33_pareto_candidates.csv" \
  "$V3_DIR/prepare/generation_conditions.json" \
  "$D0_DIR/d0_b41_table1_n20_candidates.csv" \
  "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
  "$D3_DIR/d3_event_kernel_energy.pt" \
  "$E1_DIR/e1_nl_condition_head.pt" \
  "$PREREGISTRATION" \
  "$PREDECESSOR" \
  "$E1_MANIFEST"; do
  [[ -f "$path" ]] || { echo "ERROR: missing fresh confirmation input: $path" >&2; exit 2; }
done

export PYTHONPATH="$DEP_OVERLAY:$PROJECT_DIR:$SHARED_PROJECT_DIR:$LATENT_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

case "$STAGE" in
  prepare)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/fresh_graph_jump_language_confirmation.py" \
      --protocol-manifest "$PREREGISTRATION" prepare \
      --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
      --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
      --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
      --b22-summary "$B22_DIR/summary.json" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --b36-records "$B36_DIR/train_patch_records.jsonl" \
      --trajectory-dataset "$STATE_GUIDANCE_DIR/fit_only_trajectories.pt" \
      --predecessor-manifest "$PREDECESSOR" \
      --known-source "$B26_DIR/validation_candidates.csv" \
      --known-source "$B33_DIR/fresh_internal_b33_pareto_candidates.csv" \
      --known-source "$V3_DIR/prepare/generation_conditions.json" \
      --known-source "$D0_DIR/d0_b41_table1_n20_candidates.csv" \
      --output-dir "$PREPARE_DIR"
    ;;
  freeze)
    case "$ARM_GROUP" in graph|language) ;; *) echo "ERROR: set SUCC_FRESH_CONFIRM_ARM_GROUP=graph|language" >&2; exit 2 ;; esac
    exec "$PYTHON_BIN" "$SCRIPT_DIR/fresh_graph_jump_language_confirmation.py" \
      --protocol-manifest "$PREREGISTRATION" freeze \
      --arm-group "$ARM_GROUP" \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --fit-bundle "$PREPARE_DIR/fit_support_bundle.pt" \
      --generation-conditions "$PREPARE_DIR/generation_conditions.json" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --b41-checkpoint "$B41_DIR/viability_interacting_particle_transport.pt" \
      --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
      --d3-checkpoint "$D3_DIR/d3_event_kernel_energy.pt" \
      --e1-head-checkpoint "$E1_DIR/e1_nl_condition_head.pt" \
      --e1-manifest "$E1_MANIFEST" \
      --output-dir "$OUTPUT_ROOT/frozen" \
      --device auto
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/fresh_graph_jump_language_confirmation.py" \
      --protocol-manifest "$PREREGISTRATION" evaluate \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --evaluation-targets "$PREPARE_DIR/sealed_evaluation_targets.json" \
      --frozen-root "$OUTPUT_ROOT/frozen" \
      --output-dir "$OUTPUT_ROOT/evaluation"
    ;;
  *) echo "ERROR: unsupported fresh confirmation stage: $STAGE" >&2; exit 2 ;;
esac
