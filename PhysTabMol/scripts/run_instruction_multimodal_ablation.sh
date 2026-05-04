#!/bin/bash
set -euo pipefail

# Run the same instruction-editing experiment with different multimodal contexts.
# Best used inside an interactive allocation or as a small CPU/GPU sanity run.

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
# shellcheck source=/dev/null
source "$PHYSTABMOL_ROOT/scripts/ensure_phystabmol_venv.sh"

DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
BACKEND="${PHYSTABMOL_BACKEND:-sklearn}"
MODES="${PHYSTABMOL_MULTIMODAL_MODES:-none source_image source_reference full}"
EVAL_LIMIT="${PHYSTABMOL_EVAL_LIMIT:-1000}"
SAMPLES="${PHYSTABMOL_SAMPLES:-4}"
DECODE_TOP_K="${PHYSTABMOL_DECODE_TOP_K:-1}"
RUN_PREFIX="${PHYSTABMOL_RUN_PREFIX:-instruction_multimodal}"

if [[ ! -s "$DATASET" ]]; then
  echo "Instruction dataset not found at $DATASET; building it first."
  bash scripts/build_instruction_dataset.sh
fi

if ! head -n 1 "$DATASET" | tr ',' '\n' | grep -qx 'reference_smiles'; then
  echo "Dataset has no reference_smiles column; rebuilding is recommended for source_reference/full modes." >&2
fi

EXTRA_ARGS=()
if [[ "${PHYSTABMOL_ALLOW_TARGET_REFERENCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-target-reference)
fi
if [[ "${PHYSTABMOL_DISABLE_SOURCE_AWARE_DECODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-source-aware-decoder)
fi
if [[ "${PHYSTABMOL_LATENT_VAE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--latent-vae)
fi

for mode in $MODES; do
  echo "=== multimodal_context=$mode ==="
  python3 -m phystabmol.instruction_experiment \
    --dataset "$DATASET" \
    --backend "$BACKEND" \
    --run-name "${RUN_PREFIX}_${mode}" \
    --eval-limit "$EVAL_LIMIT" \
    --samples-per-instruction "$SAMPLES" \
    --decode-top-k "$DECODE_TOP_K" \
    --multimodal-context "$mode" \
    --source-aware-pool-size "${PHYSTABMOL_SOURCE_AWARE_POOL_SIZE:-512}" \
    --source-aware-verify-candidates "${PHYSTABMOL_SOURCE_AWARE_VERIFY_CANDIDATES:-384}" \
    --torch-epochs "${PHYSTABMOL_TORCH_EPOCHS:-20}" \
    --torch-batch-size "${PHYSTABMOL_TORCH_BATCH_SIZE:-1024}" \
    --torch-hidden-dim "${PHYSTABMOL_TORCH_HIDDEN_DIM:-1024}" \
    --torch-layers "${PHYSTABMOL_TORCH_LAYERS:-6}" \
    --vae-latent-dim "${PHYSTABMOL_VAE_LATENT_DIM:-16}" \
    --vae-hidden-dim "${PHYSTABMOL_VAE_HIDDEN_DIM:-512}" \
    --vae-layers "${PHYSTABMOL_VAE_LAYERS:-3}" \
    --vae-epochs "${PHYSTABMOL_VAE_EPOCHS:-30}" \
    --vae-batch-size "${PHYSTABMOL_VAE_BATCH_SIZE:-1024}" \
    --vae-lr "${PHYSTABMOL_VAE_LR:-0.001}" \
    --vae-beta "${PHYSTABMOL_VAE_BETA:-0.001}" \
    --source-anchor-weight "${PHYSTABMOL_SOURCE_ANCHOR_WEIGHT:-0.35}" \
    --source-count-anchor-weight "${PHYSTABMOL_SOURCE_COUNT_ANCHOR_WEIGHT:-0.15}" \
    --source-anchor-neighbors "${PHYSTABMOL_SOURCE_ANCHOR_NEIGHBORS:-32}" \
    --timesteps "${PHYSTABMOL_TIMESTEPS:-40}" \
    --noise-repeats "${PHYSTABMOL_NOISE_REPEATS:-4}" \
    "${EXTRA_ARGS[@]}"
done

echo "Multimodal ablation runs finished."
