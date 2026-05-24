#!/usr/bin/env bash
# Submit a small parallel sweep of SketchImage-JEPA torch denoiser variants.
#
# This is meant for the "do not wait all night on one guess" stage: submit
# three 10GB-MIG jobs that test different hypotheses while keeping the
# SketchMol-aligned task/data defaults comparable.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_NAME="${SKETCHIMAGE_SWEEP_NAME:-sketchmol_aligned_torch_50k_10k_v7_contrastive_sweep}"
VARIANTS="${SKETCHIMAGE_SWEEP_VARIANTS:-balanced source_heavy latent_heavy}"

export SKETCHIMAGE_GPU_PROFILE="${SKETCHIMAGE_GPU_PROFILE:-h100_10gb_mig}"
export SKETCHIMAGE_MOLECULE_LIMIT="${SKETCHIMAGE_MOLECULE_LIMIT:-50000}"
export SKETCHIMAGE_MAX_TASKS="${SKETCHIMAGE_MAX_TASKS:-10000}"
export SKETCHIMAGE_TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-25}"
export SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT:-1.0}"
export SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT:-8.0}"
export SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT:-0.25}"
export SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE="${SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE:-0.10}"

echo "Submitting SketchImage-JEPA torch sweep:"
echo "  sweep_name=$SWEEP_NAME"
echo "  variants=$VARIANTS"
echo "  gpu_profile=$SKETCHIMAGE_GPU_PROFILE"
echo "  python=${SKETCHIMAGE_PYTHON_BIN:-<auto>}"
echo "  molecule_csv=${SKETCHIMAGE_MOLECULE_CSV:-<not provided>}"
echo "  molecule_limit=${SKETCHIMAGE_MOLECULE_LIMIT:-10000}"
echo "  max_tasks=${SKETCHIMAGE_MAX_TASKS:-5000}"
echo

submit_variant() {
  local variant="$1"
  shift
  (
    export SKETCHIMAGE_RUN_NAME="${SWEEP_NAME}_${variant}"
    export SKETCHIMAGE_SWEEP_VARIANT="$variant"
    for assignment in "$@"; do
      export "$assignment"
    done
    echo "=== submitting $SKETCHIMAGE_RUN_NAME ==="
    bash scripts/submit_torch_denoiser.sh
  )
}

for variant in $VARIANTS; do
  case "$variant" in
    balanced)
      submit_variant "$variant" \
        "SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT=0.05" \
        "SKETCHIMAGE_SOURCE_RERANK_WEIGHT=0.35" \
        "SKETCHIMAGE_PROPERTY_RERANK_WEIGHT=0.25" \
        "SKETCHIMAGE_SCAFFOLD_RERANK_BONUS=0.15" \
        "SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT=1.0" \
        "SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT=8.0" \
        "SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT=0.25" \
        "SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE=0.10"
      ;;
    source_heavy)
      submit_variant "$variant" \
        "SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT=0.05" \
        "SKETCHIMAGE_SOURCE_RERANK_WEIGHT=0.55" \
        "SKETCHIMAGE_PROPERTY_RERANK_WEIGHT=0.20" \
        "SKETCHIMAGE_SCAFFOLD_RERANK_BONUS=0.30" \
        "SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT=1.0" \
        "SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT=8.0" \
        "SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT=0.25" \
        "SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE=0.10"
      ;;
    latent_heavy)
      submit_variant "$variant" \
        "SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT=0.20" \
        "SKETCHIMAGE_SOURCE_RERANK_WEIGHT=0.25" \
        "SKETCHIMAGE_PROPERTY_RERANK_WEIGHT=0.20" \
        "SKETCHIMAGE_SCAFFOLD_RERANK_BONUS=0.10" \
        "SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT=2.0" \
        "SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT=12.0" \
        "SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT=0.50" \
        "SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE=0.07"
      ;;
    *)
      echo "ERROR: unknown sweep variant '$variant'." >&2
      echo "Supported variants: balanced source_heavy latent_heavy" >&2
      exit 2
      ;;
  esac
done

echo
echo "Submitted sweep variants. After jobs finish, compare with:"
echo "  SKETCHIMAGE_SWEEP_NAME=$SWEEP_NAME bash scripts/summarize_torch_sweep.sh"
