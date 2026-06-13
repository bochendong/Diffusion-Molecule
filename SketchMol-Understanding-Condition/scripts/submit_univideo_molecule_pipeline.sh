#!/usr/bin/env bash
# Submit UniVideo-style molecular generation pipeline to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"
export_smmed_source_neighbor_defaults
export_succ_edit_quality_defaults

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
TIME="${SUCC_SLURM_TIME:-24:00:00}"
MEM="${SUCC_SLURM_MEM:-80G}"
CPUS="${SUCC_SLURM_CPUS:-8}"
JOB_NAME="${SUCC_SLURM_JOB_NAME:-succ-univideo-mol}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_SLURM_PARTITION:-}"
GPU_PROFILE="${SUCC_GPU_PROFILE:-h100_40gb_mig}"
SUBMIT_MATERIALIZED_BENCHMARK="${SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER:-0}"
DEFAULT_UNIVIDEO_OUTPUT_DIR="${SUCC_DEFAULT_UNIVIDEO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_source_neighbor_v2_residual_ink}"
UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-$DEFAULT_UNIVIDEO_OUTPUT_DIR}"
STRUCTURE_BENCHMARK_DIR="${SUCC_STRUCTURE_BENCHMARK_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/image_structure_benchmark}"
DATASET_MODE="${SUCC_DATASET_MODE:-multiproperty}"

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
if [[ "$DATASET_MODE" == "moledit" || "$DATASET_MODE" == "dualmode" || "$DATASET_MODE" == "moledit_dualmode" ]]; then
  export SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=0
  export SUCC_RUN_MOLSCRIBE_OCR=0
  export SUCC_DECODE_EVAL_IMAGES="${SUCC_DECODE_EVAL_IMAGES:-0}"
fi
export SUCC_RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"
export SUCC_RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
export SUCC_UNIFIED_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  echo "       Set SUCC_PYTHON_BIN to a Python with torch, numpy, transformers/PIL for feature export, and RDKit." >&2
  exit 2
fi

if [[ -n "${SUCC_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
elif [[ "$GPU_PROFILE" == "t4" ]]; then
  GPU_CANDIDATES=("t4:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting UniVideo-style molecular generation pipeline"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  log_dir=$LOG_DIR"
echo "  output_dir=$UNIFIED_OUTPUT_DIR"
echo "  dataset_mode=$DATASET_MODE"
if [[ "$DATASET_MODE" == "moledit" || "$DATASET_MODE" == "dualmode" || "$DATASET_MODE" == "moledit_dualmode" ]]; then
  echo "  moledit_train_split=${SUCC_MOLEDIT_TRAIN_SPLIT:-${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
  echo "  moledit_eval_split=${SUCC_MOLEDIT_EVAL_SPLIT:-${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
  if [[ "$DATASET_MODE" == "dualmode" || "$DATASET_MODE" == "moledit_dualmode" ]]; then
    echo "  denovo_train_rows_per_property_count=${SUCC_DENOVO_TRAIN_ROWS_PER_PROPERTY_COUNT:-500}"
    echo "  ood_train_rows_per_spec=${SUCC_OOD_TRAIN_ROWS_PER_SPEC:-200}"
  fi
else
  echo "  condition_rows=${SUCC_CONDITION_ROWS:-$SMMED_DEFAULT_CONDITION_ROWS}"
fi
echo "  condition_features_dir=${SUCC_CONDITION_FEATURES_DIR:-$SUCC_DEFAULT_FEATURES_DIR}"
echo "  latent_backend=${SUCC_LATENT_BACKEND:-image_vae}"
echo "  latent_target_mode=${SUCC_LATENT_TARGET_MODE:-auto-residual-for-image-latents}"
echo "  residual_sample_scale=${SUCC_RESIDUAL_SAMPLE_SCALE:-1.0}"
echo "  connector_latent_blend=${SUCC_CONNECTOR_LATENT_BLEND:-0.0}"
echo "  extra_train_jsonl=${SUCC_EXTRA_TRAIN_JSONL:-none}"
echo "  extra_eval_jsonl=${SUCC_EXTRA_EVAL_JSONL:-none}"
echo "  denovo_sample_weight=${SUCC_DENOVO_SAMPLE_WEIGHT:-1.0}"
echo "  denovo_diversity_loss_weight=${SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT:-0.0}"
echo "  run_image_structure_benchmark=${SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK:-auto}"
echo "  submit_materialized_benchmark_after=$SUBMIT_MATERIALIZED_BENCHMARK"
if [[ "${SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK:-auto}" != "0" && "${SUCC_RUN_MOLSCRIBE_OCR:-auto}" != "0" ]]; then
  echo "  molscribe_model=${SUCC_MOLSCRIBE_MODEL:-${SKETCHMOL_MOLSCRIBE_MODEL:-auto-if-default-exists}}"
fi
if [[ "${SUCC_LATENT_BACKEND:-image_vae}" == "image_vae" ]]; then
  echo "  image_vae_checkpoint=${SUCC_IMAGE_VAE_CHECKPOINT:-auto-train}"
elif [[ "${SUCC_LATENT_BACKEND:-image_vae}" == "sketchmol_vae" ]]; then
  echo "  sketchmol_vae_checkpoint=${SUCC_SKETCHMOL_VAE_CHECKPOINT:-missing}"
fi

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if output="$(
    sbatch "${SBATCH_ARGS[@]}" \
      --gpus="$GPU_REQUEST" \
      --wrap="bash '$PROJECT_DIR/scripts/run_univideo_molecule_pipeline.sh'"
  )"; then
    echo "$output"
    job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit UniVideo molecule job with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi

echo
echo "Pipeline submitted."
echo "  univideo_molecule_job=$job_id"
if [[ "${SUCC_LATENT_BACKEND:-image_vae}" == "sketchmol_vae" ]]; then
  echo "  sketchmol_vae=${SUCC_SKETCHMOL_VAE_CHECKPOINT:-missing}"
elif [[ "${SUCC_LATENT_BACKEND:-image_vae}" == "image_vae" ]]; then
  echo "  image_vae=${SUCC_IMAGE_VAE_CHECKPOINT:-$UNIFIED_OUTPUT_DIR/molecule_image_vae/molecule_image_vae.pt}"
fi
echo "  metrics=$UNIFIED_OUTPUT_DIR/univideo_molecule/metrics.json"
echo "  eval=$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/metrics.json"
if [[ "${SUCC_DECODE_EVAL_IMAGES:-1}" == "1" ]]; then
  echo "  generated_images=$UNIFIED_OUTPUT_DIR/univideo_molecule/eval_latent/generated_images"
fi
if [[ "${SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK:-auto}" != "0" ]]; then
  echo "  image_structure_report=$STRUCTURE_BENCHMARK_DIR/benchmark_report.md"
fi

if [[ "$SUBMIT_MATERIALIZED_BENCHMARK" == "1" ]]; then
  echo
  echo "Submitting dependent OCR-free materialized benchmark:"
  export SUCC_MATERIALIZED_SLURM_DEPENDENCY="afterok:$job_id"
  export SUCC_UNIFIED_OUTPUT_DIR="$UNIFIED_OUTPUT_DIR"
  export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
  if [[ "$DATASET_MODE" == "moledit" ]]; then
    bash "$PROJECT_DIR/scripts/submit_univideo_moledit_benchmark.sh"
  else
    bash "$PROJECT_DIR/scripts/submit_univideo_materialized_benchmark.sh"
  fi
fi
