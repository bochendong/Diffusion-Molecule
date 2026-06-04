#!/usr/bin/env bash
# Train an image+text fusion encoder with supervised contrastive alignment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023
module load python/3.11
module load rdkit/2025.09.4

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BASELINE_CSV="${SUCC_BASELINE_CSV:-SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv}"
TARGETS_DIR="${SUCC_FUSION_TARGETS_DIR:-SketchMol-Understanding-Condition/outputs/fusion_image_text_targets_mixed_v2}"
TARGETS_NPZ="${SUCC_TARGETS_NPZ:-$TARGETS_DIR/fusion_image_text_targets.npz}"
OUTPUT_DIR="${SUCC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/fusion_image_text_encoder_mixed_v2_contrastive_e12}"
EPOCHS="${SUCC_EPOCHS:-12}"
BATCH_SIZE="${SUCC_BATCH_SIZE:-64}"
LEARNING_RATE="${SUCC_LEARNING_RATE:-1e-3}"
EMBEDDING_DIM="${SUCC_EMBEDDING_DIM:-256}"
IMAGE_SIZE="${SUCC_IMAGE_SIZE:-64}"
TEXT_DIM="${SUCC_TEXT_DIM:-256}"
CONTRASTIVE_WEIGHT="${SUCC_CONTRASTIVE_WEIGHT:-0.2}"
CONTRASTIVE_TEMPERATURE="${SUCC_CONTRASTIVE_TEMPERATURE:-0.2}"
REBUILD_TARGETS="${SUCC_REBUILD_TARGETS:-0}"

echo "Training contrastive fusion condition encoder"
echo "  baseline_csv=$BASELINE_CSV"
echo "  targets_npz=$TARGETS_NPZ"
echo "  output_dir=$OUTPUT_DIR"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  contrastive_weight=$CONTRASTIVE_WEIGHT"
echo "  contrastive_temperature=$CONTRASTIVE_TEMPERATURE"

if [[ "$REBUILD_TARGETS" == "1" || ! -f "$TARGETS_NPZ" ]]; then
  mkdir -p "$TARGETS_DIR"
  python "$PROJECT_DIR/scripts/export_fusion_image_text_targets.py" \
    --baseline-variants-csv "$BASELINE_CSV" \
    --output-dir "$TARGETS_DIR"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_fusion_image_text_encoder.py" \
  --targets-npz "$TARGETS_NPZ" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --embedding-dim "$EMBEDDING_DIM" \
  --image-size "$IMAGE_SIZE" \
  --text-dim "$TEXT_DIM" \
  --contrastive-weight "$CONTRASTIVE_WEIGHT" \
  --contrastive-temperature "$CONTRASTIVE_TEMPERATURE"

echo "Contrastive fusion condition encoder ready:"
echo "  checkpoint=$OUTPUT_DIR/fusion_image_text_encoder.pt"
echo "  metrics=$OUTPUT_DIR/metrics.json"
