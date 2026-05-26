#!/usr/bin/env bash
# Submit a paper-track ablation matrix instead of another pure hyperparameter sweep.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${SKETCHIMAGE_PAPER_MODE:-pilot}"
if [[ -n "${SKETCHIMAGE_PAPER_MATRIX_NAME:-}" ]]; then
  MATRIX_NAME="$SKETCHIMAGE_PAPER_MATRIX_NAME"
elif [[ "$MODE" == "full" ]]; then
  MATRIX_NAME="sketchmol_aligned_paper_full"
else
  MATRIX_NAME="sketchmol_aligned_paper_pilot"
fi

if [[ -n "${SKETCHIMAGE_PAPER_SEEDS:-}" ]]; then
  SEEDS="$SKETCHIMAGE_PAPER_SEEDS"
elif [[ "$MODE" == "full" ]]; then
  SEEDS="7 13 23"
else
  SEEDS="7"
fi

if [[ -n "${SKETCHIMAGE_PAPER_VARIANTS:-}" ]]; then
  VARIANTS="$SKETCHIMAGE_PAPER_VARIANTS"
elif [[ "$MODE" == "full" ]]; then
  VARIANTS="ridge_baseline planner_best planner_v2 planner_scaffold_transform planner_scaffold_transform_only planner_generative planner_generative_only planner_learned_transform planner_learned_transform_only no_contrastive weak_contrastive no_image_context"
else
  VARIANTS="ridge_baseline planner_best planner_v2 no_contrastive no_image_context"
fi

export SKETCHIMAGE_MODULES="${SKETCHIMAGE_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHIMAGE_GPU_PROFILE="${SKETCHIMAGE_GPU_PROFILE:-h100_10gb_mig}"
export SKETCHIMAGE_MOLECULE_LIMIT="${SKETCHIMAGE_MOLECULE_LIMIT:-50000}"
export SKETCHIMAGE_MAX_TASKS="${SKETCHIMAGE_MAX_TASKS:-10000}"
export SKETCHIMAGE_TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-25}"

echo "Submitting SketchImage-JEPA paper matrix:"
echo "  mode=$MODE"
echo "  matrix_name=$MATRIX_NAME"
echo "  variants=$VARIANTS"
echo "  seeds=$SEEDS"
echo "  gpu_profile=$SKETCHIMAGE_GPU_PROFILE"
echo "  modules=$SKETCHIMAGE_MODULES"
echo "  python=${SKETCHIMAGE_PYTHON_BIN:-<auto>}"
echo "  molecule_csv=${SKETCHIMAGE_MOLECULE_CSV:-<not provided>}"
echo "  molecule_limit=$SKETCHIMAGE_MOLECULE_LIMIT"
echo "  max_tasks=$SKETCHIMAGE_MAX_TASKS"
echo

submit_cpu_variant() {
  local variant="$1"
  local seed="$2"
  shift 2
  (
    export SKETCHIMAGE_RUN_NAME="${MATRIX_NAME}_${variant}_seed${seed}"
    export SKETCHIMAGE_SEED="$seed"
    export SKETCHIMAGE_BACKEND="ridge"
    export SKETCHIMAGE_RENDER_IMAGE_CONTEXT="1"
    export SKETCHIMAGE_DECODER_MODE="retrieval"
    for assignment in "$@"; do
      export "$assignment"
    done
    echo "=== submitting CPU $SKETCHIMAGE_RUN_NAME ==="
    bash scripts/submit_sketchmol_aligned.sh
  )
}

submit_gpu_variant() {
  local variant="$1"
  local seed="$2"
  shift 2
  (
    export SKETCHIMAGE_RUN_NAME="${MATRIX_NAME}_${variant}_seed${seed}"
    export SKETCHIMAGE_SEED="$seed"
    export SKETCHIMAGE_BACKEND="torch_denoiser"
    export SKETCHIMAGE_RENDER_IMAGE_CONTEXT="1"
    export SKETCHIMAGE_DECODER_MODE="retrieval"
    export SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT="0.20"
    export SKETCHIMAGE_SOURCE_RERANK_WEIGHT="0.25"
    export SKETCHIMAGE_PROPERTY_RERANK_WEIGHT="0.20"
    export SKETCHIMAGE_SCAFFOLD_RERANK_BONUS="0.10"
    export SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT="2.0"
    export SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT="12.0"
    export SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT="0.75"
    export SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE="0.04"
    for assignment in "$@"; do
      export "$assignment"
    done
    echo "=== submitting GPU $SKETCHIMAGE_RUN_NAME ==="
    bash scripts/submit_torch_denoiser.sh
  )
}

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    case "$variant" in
      ridge_baseline)
        submit_cpu_variant "$variant" "$seed"
        ;;
      planner_best)
        submit_gpu_variant "$variant" "$seed"
        ;;
      planner_v2)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_TORCH_DELTA_LOSS_WEIGHT=0.5" \
          "SKETCHIMAGE_TORCH_HARD_NEGATIVE_LOSS_WEIGHT=0.25" \
          "SKETCHIMAGE_TORCH_HARD_NEGATIVE_MARGIN=0.10"
        ;;
      planner_generative)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_DECODER_MODE=hybrid_generative" \
          "SKETCHIMAGE_GENERATIVE_SEED_COUNT=24" \
          "SKETCHIMAGE_GENERATIVE_MUTATION_ROUNDS=1" \
          "SKETCHIMAGE_GENERATIVE_CANDIDATES_PER_SEED=8" \
          "SKETCHIMAGE_GENERATIVE_NOVELTY_BONUS=0.05"
        ;;
      planner_generative_only)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_DECODER_MODE=generative" \
          "SKETCHIMAGE_GENERATIVE_SEED_COUNT=24" \
          "SKETCHIMAGE_GENERATIVE_MUTATION_ROUNDS=1" \
          "SKETCHIMAGE_GENERATIVE_CANDIDATES_PER_SEED=8" \
          "SKETCHIMAGE_GENERATIVE_NOVELTY_BONUS=0.05"
        ;;
      planner_learned_transform)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_DECODER_MODE=hybrid_learned_transform" \
          "SKETCHIMAGE_GENERATIVE_SEED_COUNT=32" \
          "SKETCHIMAGE_GENERATIVE_MUTATION_ROUNDS=1" \
          "SKETCHIMAGE_GENERATIVE_CANDIDATES_PER_SEED=8" \
          "SKETCHIMAGE_GENERATIVE_NOVELTY_BONUS=0.05"
        ;;
      planner_learned_transform_only)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_DECODER_MODE=learned_transform" \
          "SKETCHIMAGE_GENERATIVE_SEED_COUNT=32" \
          "SKETCHIMAGE_GENERATIVE_MUTATION_ROUNDS=1" \
          "SKETCHIMAGE_GENERATIVE_CANDIDATES_PER_SEED=8" \
          "SKETCHIMAGE_GENERATIVE_NOVELTY_BONUS=0.05"
        ;;
      planner_scaffold_transform)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_DECODER_MODE=hybrid_scaffold_transform" \
          "SKETCHIMAGE_GENERATIVE_SEED_COUNT=32" \
          "SKETCHIMAGE_GENERATIVE_MUTATION_ROUNDS=1" \
          "SKETCHIMAGE_GENERATIVE_CANDIDATES_PER_SEED=8" \
          "SKETCHIMAGE_GENERATIVE_NOVELTY_BONUS=0.05"
        ;;
      planner_scaffold_transform_only)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_DECODER_MODE=scaffold_transform" \
          "SKETCHIMAGE_GENERATIVE_SEED_COUNT=32" \
          "SKETCHIMAGE_GENERATIVE_MUTATION_ROUNDS=1" \
          "SKETCHIMAGE_GENERATIVE_CANDIDATES_PER_SEED=8" \
          "SKETCHIMAGE_GENERATIVE_NOVELTY_BONUS=0.05"
        ;;
      no_contrastive)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT=0.0"
        ;;
      weak_contrastive)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT=0.25" \
          "SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE=0.10"
        ;;
      no_image_context)
        submit_gpu_variant "$variant" "$seed" \
          "SKETCHIMAGE_RENDER_IMAGE_CONTEXT=0"
        ;;
      *)
        echo "ERROR: unknown paper variant '$variant'." >&2
        echo "Supported variants: ridge_baseline planner_best planner_v2 planner_scaffold_transform planner_scaffold_transform_only planner_generative planner_generative_only planner_learned_transform planner_learned_transform_only no_contrastive weak_contrastive no_image_context" >&2
        exit 2
        ;;
    esac
  done
done

echo
echo "Submitted paper matrix. After jobs finish, compare with:"
echo "  SKETCHIMAGE_PAPER_MATRIX_NAME=$MATRIX_NAME SKETCHIMAGE_PAPER_SEEDS=\"$SEEDS\" SKETCHIMAGE_PAPER_VARIANTS=\"$VARIANTS\" bash scripts/summarize_paper_matrix.sh"
