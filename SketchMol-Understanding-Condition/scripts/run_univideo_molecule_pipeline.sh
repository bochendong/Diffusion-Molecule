#!/usr/bin/env bash
# Run UniVideo-style molecular understanding + generation training locally/in a Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v1}"
FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm}"
DEFAULT_CONDITION_ROWS="SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv"
CONDITION_ROWS="${SUCC_CONDITION_ROWS:-$DEFAULT_CONDITION_ROWS}"
DATASET_DIR="${SUCC_UNIVIDEO_DATASET_DIR:-$UNIFIED_OUTPUT_DIR/dataset}"
BASELINE_CSV="${SUCC_BASELINE_CSV:-$DATASET_DIR/baseline_variants.csv}"

RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"

HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-auto}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_ATTN_IMPLEMENTATION="${SUCC_HF_ATTN_IMPLEMENTATION:-}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"

EDIT_LIMIT="${SUCC_EDIT_LIMIT:-50000}"
TRAIN_LIMIT="${SUCC_TRAIN_LIMIT:-50000}"
EVAL_LIMIT="${SUCC_EVAL_LIMIT:-1000}"
BATCH_SIZE="${SUCC_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
STAGE1_EPOCHS="${SUCC_STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${SUCC_STAGE2_EPOCHS:-5}"
STAGE3_EPOCHS="${SUCC_STAGE3_EPOCHS:-2}"
TIMESTEPS="${SUCC_TIMESTEPS:-100}"
SAMPLE_STEPS="${SUCC_SAMPLE_STEPS:-20}"
CONDITION_DROPOUT="${SUCC_CONDITION_DROPOUT:-0.1}"
SOURCE_DROPOUT="${SUCC_SOURCE_DROPOUT:-0.05}"
LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
IMAGE_SIZE="${SUCC_IMAGE_SIZE:-256}"
IMAGE_VAE_DIR="${SUCC_IMAGE_VAE_DIR:-$UNIFIED_OUTPUT_DIR/molecule_image_vae}"
IMAGE_VAE_CHECKPOINT="${SUCC_IMAGE_VAE_CHECKPOINT:-$IMAGE_VAE_DIR/molecule_image_vae.pt}"
RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-auto}"
IMAGE_VAE_EPOCHS="${SUCC_IMAGE_VAE_EPOCHS:-5}"
IMAGE_VAE_BATCH_SIZE="${SUCC_IMAGE_VAE_BATCH_SIZE:-16}"
IMAGE_VAE_LIMIT="${SUCC_IMAGE_VAE_LIMIT:-$TRAIN_LIMIT}"
IMAGE_VAE_EVAL_LIMIT="${SUCC_IMAGE_VAE_EVAL_LIMIT:-256}"
VAE_BATCH_SIZE="${SUCC_VAE_BATCH_SIZE:-16}"
DECODE_EVAL_IMAGES="${SUCC_DECODE_EVAL_IMAGES:-1}"
MAX_DECODE_IMAGES="${SUCC_MAX_DECODE_IMAGES:-$EVAL_LIMIT}"
SKETCHMOL_ROOT="${SUCC_SKETCHMOL_ROOT:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
SKETCHMOL_VAE_CONFIG="${SUCC_SKETCHMOL_VAE_CONFIG:-$SKETCHMOL_ROOT/configs/autoencoder/autoencoder_kl_pubchem400w_32x32x4.yaml}"
SKETCHMOL_VAE_CHECKPOINT="${SUCC_SKETCHMOL_VAE_CHECKPOINT:-}"
SKETCHMOL_SCALE_FACTOR="${SUCC_SKETCHMOL_SCALE_FACTOR:-1.0}"
STRUCTURE_BENCHMARK_DIR="${SUCC_STRUCTURE_BENCHMARK_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/image_structure_benchmark}"
RUN_IMAGE_STRUCTURE_BENCHMARK="${SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK:-auto}"
DEFAULT_MOLSCRIBE_MODEL="/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth"
if [[ -n "${SUCC_MOLSCRIBE_MODEL:-}" ]]; then
  MOLSCRIBE_MODEL="$SUCC_MOLSCRIBE_MODEL"
elif [[ -n "${SKETCHMOL_MOLSCRIBE_MODEL:-}" ]]; then
  MOLSCRIBE_MODEL="$SKETCHMOL_MOLSCRIBE_MODEL"
elif [[ -f "$DEFAULT_MOLSCRIBE_MODEL" ]]; then
  MOLSCRIBE_MODEL="$DEFAULT_MOLSCRIBE_MODEL"
else
  MOLSCRIBE_MODEL=""
fi
RUN_MOLSCRIBE_OCR="${SUCC_RUN_MOLSCRIBE_OCR:-auto}"
DEFAULT_MOLSCRIBE_WORKDIR=""
if [[ -d "$SKETCHMOL_ROOT/evaluate/molscribe" ]]; then
  DEFAULT_MOLSCRIBE_WORKDIR="$SKETCHMOL_ROOT/evaluate"
fi
MOLSCRIBE_WORKDIR="${SUCC_MOLSCRIBE_WORKDIR:-${SKETCHMOL_MOLSCRIBE_WORKDIR:-$DEFAULT_MOLSCRIBE_WORKDIR}}"
MOLSCRIBE_BATCH_SIZE="${SUCC_MOLSCRIBE_BATCH_SIZE:-16}"
MOLSCRIBE_DEVICE="${SUCC_MOLSCRIBE_DEVICE:-cuda}"
SOURCE_TANIMOTO_THRESHOLDS="${SUCC_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"

prepend_molscribe_pythonpath() {
  if [[ -z "$MOLSCRIBE_WORKDIR" ]]; then
    return
  fi
  if [[ -d "$MOLSCRIBE_WORKDIR/evaluate/molscribe" ]]; then
    export PYTHONPATH="$MOLSCRIBE_WORKDIR/evaluate:$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  else
    export PYTHONPATH="$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

check_molscribe_import() {
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg  # noqa: F401
    from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights  # noqa: F401
    from molscribe import MolScribe  # noqa: F401
except Exception as exc:
    print("ERROR: MolScribe/timm compatibility check failed:", file=sys.stderr)
    print(f"  {exc}", file=sys.stderr)
    print("Hint: set SUCC_MOLSCRIBE_WORKDIR to a MolScribe checkout/evaluate directory,", file=sys.stderr)
    print("      or install a compatible MolScribe + timm into SUCC_PYTHON_BIN.", file=sys.stderr)
    print(f"      {sys.executable} -m pip install --force-reinstall --no-deps timm==0.4.12", file=sys.stderr)
    sys.exit(2)
PY
}

echo "Running UniVideo-style molecular generation pipeline"
echo "  python=$PYTHON_BIN"
echo "  condition_rows=$CONDITION_ROWS"
echo "  baseline_csv=$BASELINE_CSV"
echo "  condition_features_dir=$FEATURES_DIR"
echo "  unified_output_dir=$UNIFIED_OUTPUT_DIR"
echo "  dataset_dir=$DATASET_DIR"
echo "  run_dataset_export=$RUN_DATASET_EXPORT"
echo "  run_feature_export=$RUN_FEATURE_EXPORT"
echo "  latent_backend=$LATENT_BACKEND"
echo "  max_decode_images=$MAX_DECODE_IMAGES"
echo "  run_image_structure_benchmark=$RUN_IMAGE_STRUCTURE_BENCHMARK"
echo "  run_molscribe_ocr=$RUN_MOLSCRIBE_OCR"
echo "  molscribe_model=${MOLSCRIBE_MODEL:-missing}"
echo "  molscribe_workdir=${MOLSCRIBE_WORKDIR:-pythonpath-default}"
if [[ "$LATENT_BACKEND" == "image_vae" ]]; then
  echo "  image_vae_checkpoint=$IMAGE_VAE_CHECKPOINT"
  echo "  run_image_vae_train=$RUN_IMAGE_VAE_TRAIN"
elif [[ "$LATENT_BACKEND" == "sketchmol_vae" ]]; then
  echo "  sketchmol_root=$SKETCHMOL_ROOT"
  echo "  sketchmol_vae_config=$SKETCHMOL_VAE_CONFIG"
  echo "  sketchmol_vae_checkpoint=${SKETCHMOL_VAE_CHECKPOINT:-missing}"
fi

if [[ "$RUN_DATASET_EXPORT" == "1" || ! -f "$DATASET_DIR/univideo_edit_train.jsonl" || ! -f "$DATASET_DIR/univideo_edit_eval.jsonl" || ! -f "$BASELINE_CSV" ]]; then
  if [[ ! -f "$CONDITION_ROWS" ]]; then
    echo "ERROR: missing condition rows CSV: $CONDITION_ROWS" >&2
    echo "Set SUCC_CONDITION_ROWS to the condition_rows.csv produced by your dataset build." >&2
    exit 2
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_univideo_edit_dataset.py" \
    --condition-rows-csv "$CONDITION_ROWS" \
    --output-dir "$DATASET_DIR" \
    --limit "$EDIT_LIMIT" \
    --variants full
fi

if [[ "$RUN_FEATURE_EXPORT" == "1" || ( "$RUN_FEATURE_EXPORT" == "auto" && ! -f "$FEATURES_DIR/query_tokens.npy" ) ]]; then
  if [[ -z "$HF_MODEL_NAME_OR_PATH" ]]; then
    echo "ERROR: SUCC_HF_MODEL_NAME_OR_PATH is required to export HF VLM features." >&2
    echo "Set SUCC_RUN_FEATURE_EXPORT=0 if $FEATURES_DIR already exists." >&2
    exit 2
  fi
  export SUCC_ENCODER=hf_vlm
  export SUCC_VARIANTS=full
  export SUCC_BASELINE_CSV="$BASELINE_CSV"
  export SUCC_OUTPUT_DIR="$FEATURES_DIR"
  export SUCC_POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
  export SUCC_NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
  export SUCC_QUERY_DIM="${SUCC_QUERY_DIM:-256}"
  export SUCC_HF_MODEL_NAME_OR_PATH="$HF_MODEL_NAME_OR_PATH"
  export SUCC_HF_DEVICE_MAP="$HF_DEVICE_MAP"
  export SUCC_HF_DTYPE="$HF_DTYPE"
  export SUCC_HF_BATCH_SIZE="$HF_BATCH_SIZE"
  export SUCC_HF_MAX_LENGTH="$HF_MAX_LENGTH"
  export SUCC_HF_RENDER_IMAGE_SIZE="$HF_RENDER_IMAGE_SIZE"
  if [[ -n "$HF_ATTN_IMPLEMENTATION" ]]; then
    export SUCC_HF_ATTN_IMPLEMENTATION="$HF_ATTN_IMPLEMENTATION"
  fi
  bash "$PROJECT_DIR/scripts/run_condition_encoder_export.sh"
fi

mkdir -p "$UNIFIED_OUTPUT_DIR"

LATENT_BACKEND_ARGS=()
if [[ "$LATENT_BACKEND" == "image_vae" ]]; then
  if [[ "$RUN_IMAGE_VAE_TRAIN" == "1" || ( "$RUN_IMAGE_VAE_TRAIN" == "auto" && ! -f "$IMAGE_VAE_CHECKPOINT" ) ]]; then
    echo "Training molecule-image VAE for SketchMol-style 4x32x32 latents"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_molecule_image_vae.py" \
      --train-jsonl "$DATASET_DIR/univideo_edit_train.jsonl" \
      --eval-jsonl "$DATASET_DIR/univideo_edit_eval.jsonl" \
      --output-dir "$IMAGE_VAE_DIR" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$IMAGE_VAE_BATCH_SIZE" \
      --epochs "$IMAGE_VAE_EPOCHS" \
      --limit "$IMAGE_VAE_LIMIT" \
      --eval-limit "$IMAGE_VAE_EVAL_LIMIT"
  fi
  if [[ ! -f "$IMAGE_VAE_CHECKPOINT" ]]; then
    echo "ERROR: missing image VAE checkpoint: $IMAGE_VAE_CHECKPOINT" >&2
    echo "Set SUCC_IMAGE_VAE_CHECKPOINT, or allow SUCC_RUN_IMAGE_VAE_TRAIN=auto/1 to create it." >&2
    exit 2
  fi
  LATENT_BACKEND_ARGS=(
    --latent-backend image_vae
    --image-vae-checkpoint "$IMAGE_VAE_CHECKPOINT"
    --image-size "$IMAGE_SIZE"
    --vae-batch-size "$VAE_BATCH_SIZE"
    --max-decode-images "$MAX_DECODE_IMAGES"
  )
  if [[ "$DECODE_EVAL_IMAGES" == "1" ]]; then
    LATENT_BACKEND_ARGS+=(--decode-eval-images)
  fi
elif [[ "$LATENT_BACKEND" == "fingerprint_property_vector" ]]; then
  LATENT_BACKEND_ARGS=(--latent-backend fingerprint_property_vector)
elif [[ "$LATENT_BACKEND" == "sketchmol_vae" ]]; then
  if [[ -z "$SKETCHMOL_VAE_CHECKPOINT" ]]; then
    echo "ERROR: SUCC_SKETCHMOL_VAE_CHECKPOINT is required when SUCC_LATENT_BACKEND=sketchmol_vae" >&2
    exit 2
  fi
  if [[ ! -d "$SKETCHMOL_ROOT" ]]; then
    echo "ERROR: missing SketchMol root: $SKETCHMOL_ROOT" >&2
    exit 2
  fi
  if [[ ! -f "$SKETCHMOL_VAE_CONFIG" ]]; then
    echo "ERROR: missing SketchMol VAE config: $SKETCHMOL_VAE_CONFIG" >&2
    exit 2
  fi
  if [[ ! -f "$SKETCHMOL_VAE_CHECKPOINT" ]]; then
    echo "ERROR: missing SketchMol VAE checkpoint: $SKETCHMOL_VAE_CHECKPOINT" >&2
    exit 2
  fi
  SKETCHMOL_ROOT_FOR_CHECK="$SKETCHMOL_ROOT" "$PYTHON_BIN" - <<'PY'
import os
import sys

root = os.environ["SKETCHMOL_ROOT_FOR_CHECK"]
if root not in sys.path:
    sys.path.insert(0, root)

import omegaconf  # noqa: F401
import ldm.models.autoencoder  # noqa: F401
PY
  LATENT_BACKEND_ARGS=(
    --latent-backend sketchmol_vae
    --sketchmol-root "$SKETCHMOL_ROOT"
    --sketchmol-vae-config "$SKETCHMOL_VAE_CONFIG"
    --sketchmol-vae-checkpoint "$SKETCHMOL_VAE_CHECKPOINT"
    --sketchmol-scale-factor "$SKETCHMOL_SCALE_FACTOR"
    --image-size "$IMAGE_SIZE"
    --vae-batch-size "$VAE_BATCH_SIZE"
    --max-decode-images "$MAX_DECODE_IMAGES"
  )
  if [[ "$DECODE_EVAL_IMAGES" == "1" ]]; then
    LATENT_BACKEND_ARGS+=(--decode-eval-images)
  fi
else
  echo "ERROR: unsupported SUCC_LATENT_BACKEND=$LATENT_BACKEND" >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_univideo_molecule_generation.py" \
  --train-jsonl "$DATASET_DIR/univideo_edit_train.jsonl" \
  --eval-jsonl "$DATASET_DIR/univideo_edit_eval.jsonl" \
  --condition-features-dir "$FEATURES_DIR" \
  --condition-feature-array query_tokens \
  --condition-feature-variant full \
  --output-dir "$UNIFIED_OUTPUT_DIR/univideo_molecule" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --stage1-epochs "$STAGE1_EPOCHS" \
  --stage2-epochs "$STAGE2_EPOCHS" \
  --stage3-epochs "$STAGE3_EPOCHS" \
  --timesteps "$TIMESTEPS" \
  --limit "$TRAIN_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --sample-steps "$SAMPLE_STEPS" \
  --condition-dropout "$CONDITION_DROPOUT" \
  --source-dropout "$SOURCE_DROPOUT" \
  "${LATENT_BACKEND_ARGS[@]}" \
  --export-condition-tokens

if [[ "$LATENT_BACKEND" != "fingerprint_property_vector" && "$DECODE_EVAL_IMAGES" == "1" ]]; then
  IMAGE_CSV="$STRUCTURE_BENCHMARK_DIR/image_path.csv"
  GENERATED_IMAGES_DIR="$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/generated_images"
  PREDICTIONS_CSV="$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/predictions.csv"
  if [[ "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "1" || "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "auto" || "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "prepare" ]]; then
    echo "Preparing MolScribe image CSV for image-to-structure benchmark"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_univideo_molscribe_csv.py" \
      --predictions-csv "$PREDICTIONS_CSV" \
      --eval-jsonl "$DATASET_DIR/univideo_edit_eval.jsonl" \
      --generated-images-dir "$GENERATED_IMAGES_DIR" \
      --output-csv "$IMAGE_CSV" \
      --method "univideo_${LATENT_BACKEND}"
  fi
  if [[ "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "prepare" ]]; then
    echo "Prepared image CSV only; skipping MolScribe OCR/evaluation because SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=prepare"
  elif [[ "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "1" || "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "auto" ]]; then
    if [[ "$RUN_MOLSCRIBE_OCR" == "0" ]]; then
      echo "Skipping MolScribe OCR/evaluation because SUCC_RUN_MOLSCRIBE_OCR=0"
    else
      if [[ -z "$MOLSCRIBE_MODEL" ]]; then
        if [[ "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "1" ]]; then
          echo "ERROR: MolScribe model missing. Set SUCC_MOLSCRIBE_MODEL or SKETCHMOL_MOLSCRIBE_MODEL." >&2
          exit 2
        fi
        echo "Skipping MolScribe OCR/evaluation because no MolScribe checkpoint was found."
      else
        echo "Running MolScribe OCR and image-to-structure benchmark"
        prepend_molscribe_pythonpath
        if ! check_molscribe_import; then
          if [[ "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "1" || "$RUN_MOLSCRIBE_OCR" == "1" ]]; then
            exit 2
          fi
          echo "Skipping MolScribe OCR/evaluation because MolScribe is unavailable in auto mode."
          echo "Set SUCC_MOLSCRIBE_WORKDIR=$SKETCHMOL_ROOT/evaluate or install MolScribe into $PYTHON_BIN."
        else
          "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_molscribe_ocr.py" \
            --model-path "$MOLSCRIBE_MODEL" \
            --image-csv "$IMAGE_CSV" \
            --batch-size "$MOLSCRIBE_BATCH_SIZE" \
            --device "$MOLSCRIBE_DEVICE"
          "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
            --image-csv "$IMAGE_CSV" \
            --output-dir "$STRUCTURE_BENCHMARK_DIR" \
            --method "univideo_${LATENT_BACKEND}" \
            --source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS"
        fi
      fi
    fi
  fi
elif [[ "$RUN_IMAGE_STRUCTURE_BENCHMARK" == "1" ]]; then
  echo "ERROR: image-to-structure benchmark requires image/sketchmol VAE backend with SUCC_DECODE_EVAL_IMAGES=1." >&2
  exit 2
fi

echo "UniVideo-style molecular pipeline finished:"
echo "  dataset=$DATASET_DIR/summary.json"
if [[ "$LATENT_BACKEND" == "image_vae" ]]; then
  echo "  image_vae=$IMAGE_VAE_CHECKPOINT"
  echo "  generated_images=$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/generated_images"
elif [[ "$LATENT_BACKEND" == "sketchmol_vae" ]]; then
  echo "  sketchmol_vae=$SKETCHMOL_VAE_CHECKPOINT"
  echo "  generated_images=$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/generated_images"
fi
echo "  checkpoint=$UNIFIED_OUTPUT_DIR/univideo_molecule/univideo_molecule_generation.pt"
echo "  metrics=$UNIFIED_OUTPUT_DIR/univideo_molecule/metrics.json"
echo "  eval=$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/metrics.json"
echo "  condition_tokens=$UNIFIED_OUTPUT_DIR/univideo_molecule/condition_tokens_train/query_tokens.npy"
if [[ "$LATENT_BACKEND" != "fingerprint_property_vector" && "$DECODE_EVAL_IMAGES" == "1" ]]; then
  echo "  image_structure_csv=$STRUCTURE_BENCHMARK_DIR/image_path.csv"
  echo "  image_structure_report=$STRUCTURE_BENCHMARK_DIR/benchmark_report.md"
fi
