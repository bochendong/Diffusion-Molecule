#!/usr/bin/env bash
# Run one preregistered language-grounded transport-adapter arm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_LANG_FLOW_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_LANG_FLOW_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
ARM="${SUCC_LANG_FLOW_ARM:?Set SUCC_LANG_FLOW_ARM}"
SEED="${SUCC_LANG_FLOW_SEED:-2001}"
OUTPUT_ROOT="${SUCC_LANG_FLOW_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/language_grounded_graph_latent_flow_v1/seed_${SEED}}"
OUTPUT_DIR="$OUTPUT_ROOT/$ARM"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B36_DIR="$SHARED_PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
B37_DIR="$SHARED_PROJECT_DIR/outputs/source_clamped_region_graph_diffusion_v37/seed_1983"
B38_DIR="$SHARED_PROJECT_DIR/outputs/source_clamped_latent_graph_jump_process_v38/seed_1985"
B39_DIR="$SHARED_PROJECT_DIR/outputs/latent_cardinality_graph_jump_bridge_v39/seed_1987"
B40_DIR="$SHARED_PROJECT_DIR/outputs/valence_constrained_latent_particle_bridge_v40/seed_1989"
B41_DIR="$SHARED_PROJECT_DIR/outputs/viability_preserving_interacting_particle_transport_v41/seed_1991"
VALID_TERMINAL_DIR="$SHARED_PROJECT_DIR/outputs/valid_terminal_molecule_latent_jump_v1/seed_1991"
STATE_GUIDANCE_DIR="$SHARED_PROJECT_DIR/outputs/common_llm_state_viability_guidance_v1/seed_1995"
SFT_ADAPTER_DIR="${SUCC_LANG_FLOW_SFT_ADAPTER_DIR:-$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter}"
PREREGISTRATION="$SCRIPT_DIR/language_grounded_graph_latent_flow_v1_preregistration.json"

case "$ARM" in
  property_memory|common_llm_memory) ;;
  *) echo "ERROR: unsupported language-grounded flow arm: $ARM" >&2; exit 2 ;;
esac

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json" \
  "$B36_DIR/summary.json" \
  "$B37_DIR/summary.json" \
  "$B38_DIR/source_clamped_latent_graph_jump_process.pt" \
  "$B38_DIR/summary.json" \
  "$B39_DIR/latent_cardinality_graph_jump_bridge.pt" \
  "$B39_DIR/summary.json" \
  "$B39_DIR/evaluated_train_only_dev_candidates.csv" \
  "$B40_DIR/summary.json" \
  "$B40_DIR/evaluated_train_only_dev_candidates.csv" \
  "$B41_DIR/viability_interacting_particle_transport.pt" \
  "$B41_DIR/summary.json" \
  "$B41_DIR/evaluated_train_only_dev_candidates.csv" \
  "$VALID_TERMINAL_DIR/summary.json" \
  "$VALID_TERMINAL_DIR/evaluated_train_only_dev_candidates.csv" \
  "$STATE_GUIDANCE_DIR/merged/summary.json" \
  "$STATE_GUIDANCE_DIR/prepare/fit_only_trajectories.pt" \
  "$SFT_ADAPTER_DIR/adapter_config.json" \
  "$SFT_ADAPTER_DIR/adapter_model.safetensors" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing language-grounded input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_LANG_FLOW_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed language-grounded arm exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$DEP_OVERLAY:$PROJECT_DIR:$SHARED_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/language_grounded_graph_latent_flow.py" \
  --arm "$ARM" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  --b22-summary "$B22_DIR/summary.json" \
  --b36-summary "$B36_DIR/summary.json" \
  --b37-summary "$B37_DIR/summary.json" \
  --b38-checkpoint "$B38_DIR/source_clamped_latent_graph_jump_process.pt" \
  --b38-summary "$B38_DIR/summary.json" \
  --b39-checkpoint "$B39_DIR/latent_cardinality_graph_jump_bridge.pt" \
  --b39-summary "$B39_DIR/summary.json" \
  --b39-evaluated-candidates "$B39_DIR/evaluated_train_only_dev_candidates.csv" \
  --b40-summary "$B40_DIR/summary.json" \
  --b40-evaluated-candidates "$B40_DIR/evaluated_train_only_dev_candidates.csv" \
  --b41-checkpoint "$B41_DIR/viability_interacting_particle_transport.pt" \
  --b41-summary "$B41_DIR/summary.json" \
  --b41-evaluated-candidates "$B41_DIR/evaluated_train_only_dev_candidates.csv" \
  --valid-terminal-summary "$VALID_TERMINAL_DIR/summary.json" \
  --valid-terminal-candidates "$VALID_TERMINAL_DIR/evaluated_train_only_dev_candidates.csv" \
  --state-guidance-summary "$STATE_GUIDANCE_DIR/merged/summary.json" \
  --trajectory-dataset "$STATE_GUIDANCE_DIR/prepare/fit_only_trajectories.pt" \
  --sft-adapter-dir "$SFT_ADAPTER_DIR" \
  --protocol-manifest "$PREREGISTRATION" \
  --output-dir "$OUTPUT_DIR" \
  --device auto
