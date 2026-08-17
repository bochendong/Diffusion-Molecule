#!/usr/bin/env bash
# Run one stage of the direction-only prospective latent-flow confirmation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_LANG_FRESH_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_LANG_FRESH_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_LANG_FRESH_STAGE:?Set SUCC_LANG_FRESH_STAGE}"
ARM="${SUCC_LANG_FRESH_ARM:-}"
OUTPUT_ROOT="${SUCC_LANG_FRESH_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/language_grounded_graph_latent_fresh_v2/seed_2011}"
PREPARE_DIR="$OUTPUT_ROOT/prepare"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B41_DIR="$SHARED_PROJECT_DIR/outputs/viability_preserving_interacting_particle_transport_v41/seed_1991"
B26_DIR="$SHARED_PROJECT_DIR/outputs/frozen_fragment_attachment_fresh_holdout_v26/seed_1873"
B33_DIR="$SHARED_PROJECT_DIR/outputs/pareto_conditioned_joint_latent_v33/seed_1951"
SFT_ADAPTER_DIR="${SUCC_LANG_FRESH_SFT_ADAPTER_DIR:-$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter}"
PREREGISTRATION="$SCRIPT_DIR/language_grounded_graph_latent_fresh_v2_preregistration.json"
PREDECESSOR="$SCRIPT_DIR/language_grounded_graph_latent_flow_v1_preregistration.json"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json" \
  "$B41_DIR/viability_interacting_particle_transport.pt" \
  "$B26_DIR/validation_candidates.csv" \
  "$B33_DIR/fresh_internal_b33_pareto_candidates.csv" \
  "$SFT_ADAPTER_DIR/adapter_config.json" \
  "$SFT_ADAPTER_DIR/adapter_model.safetensors" \
  "$PREREGISTRATION" \
  "$PREDECESSOR"; do
  [[ -f "$path" ]] || { echo "ERROR: missing direction-only fresh input: $path" >&2; exit 2; }
done

export PYTHONPATH="$DEP_OVERLAY:$PROJECT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

case "$STAGE" in
  prepare)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/language_grounded_graph_latent_fresh_confirmation.py" \
      --protocol-manifest "$PREREGISTRATION" prepare \
      --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
      --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
      --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
      --b22-summary "$B22_DIR/summary.json" \
      --predecessor-manifest "$PREDECESSOR" \
      --known-source-csv "$B26_DIR/validation_candidates.csv" \
      --known-source-csv "$B33_DIR/fresh_internal_b33_pareto_candidates.csv" \
      --output-dir "$PREPARE_DIR"
    ;;
  arm)
    case "$ARM" in property_memory|common_llm_memory) ;; *) echo "ERROR: invalid arm: $ARM" >&2; exit 2 ;; esac
    exec "$PYTHON_BIN" "$SCRIPT_DIR/language_grounded_graph_latent_fresh_confirmation.py" \
      --protocol-manifest "$PREREGISTRATION" arm \
      --arm "$ARM" \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --fit-bundle "$PREPARE_DIR/fit_only_direction_pairs.pt" \
      --generation-conditions "$PREPARE_DIR/generation_conditions.json" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --b41-checkpoint "$B41_DIR/viability_interacting_particle_transport.pt" \
      --sft-adapter-dir "$SFT_ADAPTER_DIR" \
      --output-dir "$OUTPUT_ROOT/$ARM" \
      --device auto
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/language_grounded_graph_latent_fresh_confirmation.py" \
      --protocol-manifest "$PREREGISTRATION" evaluate \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --evaluation-targets "$PREPARE_DIR/sealed_evaluation_targets.json" \
      --property-memory-summary "$OUTPUT_ROOT/property_memory/summary.json" \
      --property-memory-candidates "$OUTPUT_ROOT/property_memory/frozen_prospective_candidates.csv" \
      --common-llm-memory-summary "$OUTPUT_ROOT/common_llm_memory/summary.json" \
      --common-llm-memory-candidates "$OUTPUT_ROOT/common_llm_memory/frozen_prospective_candidates.csv" \
      --output-dir "$OUTPUT_ROOT/evaluation"
    ;;
  *) echo "ERROR: unsupported direction-only fresh stage: $STAGE" >&2; exit 2 ;;
esac
