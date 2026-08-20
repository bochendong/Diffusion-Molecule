#!/usr/bin/env bash
# Run one physically separated stage of the semantic-energy graph-jump pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_SEMANTIC_ENERGY_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_SEMANTIC_ENERGY_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
STAGE="${SUCC_SEMANTIC_ENERGY_STAGE:?Set SUCC_SEMANTIC_ENERGY_STAGE}"
OUTPUT_ROOT="${SUCC_SEMANTIC_ENERGY_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/semantic_energy_graph_jump_v1/seed_2045}"
PREPARE_DIR="$OUTPUT_ROOT/prepare"
TRAIN_DIR="$OUTPUT_ROOT/train"
FROZEN_DIR="$OUTPUT_ROOT/frozen"
EVALUATION_DIR="$OUTPUT_ROOT/evaluation"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
CANONICAL_DIR="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical"
PREDECESSOR_BUNDLE="$SHARED_PROJECT_DIR/outputs/language_grounded_graph_latent_fresh_edit_v3/seed_2022/prepare/fit_only_direction_pairs.pt"
SFT_ADAPTER_DIR="${SUCC_SEMANTIC_ENERGY_SFT_ADAPTER_DIR:-$SHARED_PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter}"
PROTOCOL="$SCRIPT_DIR/semantic_energy_graph_jump_v1_preregistration.json"
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

for path in "$PROTOCOL" "$E1_MANIFEST" "$PREDECESSOR_BUNDLE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing semantic-energy input: $path" >&2; exit 2; }
done

case "$STAGE" in
  prepare)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/semantic_energy_graph_jump_v1.py" \
      --protocol-manifest "$PROTOCOL" prepare \
      --predecessor-fit-bundle "$PREDECESSOR_BUNDLE" \
      --e1-manifest "$E1_MANIFEST" \
      --output-dir "$PREPARE_DIR"
    ;;
  train)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/semantic_energy_graph_jump_v1.py" \
      --protocol-manifest "$PROTOCOL" train \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --fit-probe-bundle "$PREPARE_DIR/fit_probe_bundle.pt" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
      --sft-adapter-dir "$SFT_ADAPTER_DIR" \
      --e1-manifest "$E1_MANIFEST" \
      --output-dir "$TRAIN_DIR" \
      --device auto
    ;;
  freeze)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/semantic_energy_graph_jump_v1.py" \
      --protocol-manifest "$PROTOCOL" freeze \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --fit-probe-bundle "$PREPARE_DIR/fit_probe_bundle.pt" \
      --generation-conditions "$PREPARE_DIR/generation_conditions.json" \
      --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
      --representation-summary "$REPRESENTATION_DIR/summary.json" \
      --canonical-checkpoint "$CANONICAL_DIR/b41_canonical_event_kernel.pt" \
      --sft-adapter-dir "$SFT_ADAPTER_DIR" \
      --adapter-checkpoint "$TRAIN_DIR/semantic_energy_adapter.pt" \
      --train-summary "$TRAIN_DIR/summary.json" \
      --output-dir "$FROZEN_DIR" \
      --device auto
    ;;
  evaluate)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/semantic_energy_graph_jump_v1.py" \
      --protocol-manifest "$PROTOCOL" evaluate \
      --prepare-summary "$PREPARE_DIR/summary.json" \
      --evaluation-targets "$PREPARE_DIR/sealed_evaluation_targets.json" \
      --frozen-root "$FROZEN_DIR" \
      --output-dir "$EVALUATION_DIR"
    ;;
  *) echo "ERROR: unsupported semantic-energy stage: $STAGE" >&2; exit 2 ;;
esac
