#!/usr/bin/env bash
# One-click Slurm submission for the inode-safe dataset build plus big-VLM benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATASET_PROJECT_DIR="$REPO_DIR/SketchMol-MultiProperty-EditDataset"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

DATASET_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
MOLECULE_DB="${SMMED_MOLECULE_DB_CSV:-$DATASET_OUTPUT_DIR/molecule_database.csv}"
FEATURES_DIR="${SUCC_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm}"
BENCHMARK_OUTPUT_DIR="${SMMED_BENCHMARK_OUTPUT_DIR:-$DATASET_OUTPUT_DIR/benchmark_hf_vlm}"
CONNECTOR_FEATURES_DIR="${SUCC_CONNECTOR_FEATURES_DIR:-${FEATURES_DIR}_edit_connector}"
CONNECTOR_BENCHMARK_OUTPUT_DIR="${SMMED_CONNECTOR_BENCHMARK_OUTPUT_DIR:-${BENCHMARK_OUTPUT_DIR}_edit_connector}"
SUBMIT_DATASET_BUILD="${SMMED_SUBMIT_DATASET_BUILD:-1}"

export SMMED_RENDER_IMAGES="${SMMED_RENDER_IMAGES:-0}"
export SMMED_OUTPUT_DIR="$DATASET_OUTPUT_DIR"
export SMMED_MOLECULE_DB_CSV="$MOLECULE_DB"
export SUCC_FEATURES_DIR="$FEATURES_DIR"
export SMMED_BENCHMARK_OUTPUT_DIR="$BENCHMARK_OUTPUT_DIR"
export SUCC_CONNECTOR_FEATURES_DIR="$CONNECTOR_FEATURES_DIR"
export SMMED_CONNECTOR_BENCHMARK_OUTPUT_DIR="$CONNECTOR_BENCHMARK_OUTPUT_DIR"
export SUCC_TRAIN_FEATURE_CONNECTOR="${SUCC_TRAIN_FEATURE_CONNECTOR:-1}"
export SMMED_SCAFFOLD_FALLBACK_MODE="${SMMED_SCAFFOLD_FALLBACK_MODE:-source_identity}"
export SMMED_RERANK_PROPERTY_WEIGHT="${SMMED_RERANK_PROPERTY_WEIGHT:-0.5}"
export SMMED_RERANK_CANDIDATES="${SMMED_RERANK_CANDIDATES:-64}"
export SMMED_EDIT_BENCHMARK_METHODS="${SMMED_EDIT_BENCHMARK_METHODS:-source_identity,global_property_retrieval,scaffold_property_retrieval,edit_latent_global_retrieval,edit_latent_scaffold_retrieval,edit_latent_scaffold_source_rerank,target_oracle}"
export SMMED_MAX_EDIT_LATENT_CANDIDATES="${SMMED_MAX_EDIT_LATENT_CANDIDATES:-20000}"
export SMMED_EDIT_LATENT_PROPERTY_WEIGHT="${SMMED_EDIT_LATENT_PROPERTY_WEIGHT:-1.0}"
export SMMED_EDIT_LATENT_DELTA_WEIGHT="${SMMED_EDIT_LATENT_DELTA_WEIGHT:-0.35}"
export SMMED_EDIT_LATENT_DIRECTION_WEIGHT="${SMMED_EDIT_LATENT_DIRECTION_WEIGHT:-0.10}"
export SMMED_EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT="${SMMED_EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT:-0.25}"

BUILD_ACCOUNT="${SMMED_SLURM_ACCOUNT:-def-hup-ab_gpu}"
BUILD_TIME="${SMMED_SLURM_TIME:-08:00:00}"
BUILD_MEM="${SMMED_SLURM_MEM:-32G}"
BUILD_CPUS="${SMMED_SLURM_CPUS:-4}"
BUILD_JOB_NAME="${SMMED_SLURM_JOB_NAME:-smmed-build}"
BUILD_LOG_DIR="${SMMED_LOG_DIR:-$DATASET_PROJECT_DIR/logs}"

VLM_ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
VLM_TIME="${SUCC_SLURM_TIME:-24:00:00}"
VLM_MEM="${SUCC_SLURM_MEM:-80G}"
VLM_CPUS="${SUCC_SLURM_CPUS:-8}"
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
VLM_GPU_PROFILE="${SUCC_GPU_PROFILE:-h100_40gb_mig}"
VLM_JOB_NAME="${SUCC_SLURM_JOB_NAME:-succ-hf-vlm}"
VLM_LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
VLM_PARTITION="${SUCC_SLURM_PARTITION:-}"
HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

mkdir -p "$BUILD_LOG_DIR" "$VLM_LOG_DIR"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  echo "       Set SUCC_PYTHON_BIN to a VLM/RDKit Python with torch, transformers, PIL, and RDKit." >&2
  exit 2
fi
if [[ "$HF_MODEL_NAME_OR_PATH" == /* && ! -e "$HF_MODEL_NAME_OR_PATH" ]]; then
  echo "ERROR: local HF model path does not exist: $HF_MODEL_NAME_OR_PATH" >&2
  echo "       Use an existing checkpoint path or a HuggingFace repo id." >&2
  exit 2
fi

if [[ -n "${SUCC_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_SLURM_GPUS")
elif [[ "$VLM_GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$VLM_GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$VLM_GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$VLM_GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$VLM_GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
elif [[ "$VLM_GPU_PROFILE" == "t4" ]]; then
  GPU_CANDIDATES=("t4:1")
else
  GPU_CANDIDATES=("$VLM_GPU_PROFILE")
fi

echo "Submitting inode-safe HF VLM multi-property pipeline"
echo "  dataset_output_dir=$DATASET_OUTPUT_DIR"
echo "  molecule_db=$MOLECULE_DB"
echo "  features_dir=$FEATURES_DIR"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  connector_features_dir=$CONNECTOR_FEATURES_DIR"
echo "  connector_benchmark_output_dir=$CONNECTOR_BENCHMARK_OUTPUT_DIR"
echo "  render_images=$SMMED_RENDER_IMAGES"
echo "  model=$HF_MODEL_NAME_OR_PATH"
echo "  python=$SUCC_PYTHON_BIN"
echo "  train_feature_connector=$SUCC_TRAIN_FEATURE_CONNECTOR"
echo "  scaffold_fallback_mode=$SMMED_SCAFFOLD_FALLBACK_MODE"
echo "  rerank_candidates=$SMMED_RERANK_CANDIDATES"
echo "  rerank_property_weight=$SMMED_RERANK_PROPERTY_WEIGHT"
echo "  edit_methods=$SMMED_EDIT_BENCHMARK_METHODS"
echo "  edit_latent_source_similarity_weight=$SMMED_EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT"
echo "  gpu_profile=$VLM_GPU_PROFILE"
echo "  slurm_gpu_candidates=${GPU_CANDIDATES[*]}"

build_job_id=""
if [[ "$SUBMIT_DATASET_BUILD" == "1" ]]; then
  echo
  echo "Submitting dataset build job:"
  echo "  account=$BUILD_ACCOUNT"
  echo "  time=$BUILD_TIME"
  echo "  mem=$BUILD_MEM"
  echo "  cpus=$BUILD_CPUS"
  echo "  log_dir=$BUILD_LOG_DIR"
  build_output="$(
    sbatch \
      --account="$BUILD_ACCOUNT" \
      --job-name="$BUILD_JOB_NAME" \
      --time="$BUILD_TIME" \
      --mem="$BUILD_MEM" \
      --cpus-per-task="$BUILD_CPUS" \
      --output="$BUILD_LOG_DIR/%x-%j.log" \
      --export=ALL \
      --wrap="bash '$DATASET_PROJECT_DIR/scripts/run_build_dataset.sh'"
  )"
  echo "$build_output"
  build_job_id="$(echo "$build_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$build_job_id" ]]; then
    echo "ERROR: could not parse dataset build job id from sbatch output." >&2
    exit 1
  fi
else
  echo
  echo "Skipping dataset build submission because SMMED_SUBMIT_DATASET_BUILD=$SUBMIT_DATASET_BUILD"
fi

VLM_SBATCH_ARGS=(
  --account="$VLM_ACCOUNT"
  --job-name="$VLM_JOB_NAME"
  --time="$VLM_TIME"
  --mem="$VLM_MEM"
  --cpus-per-task="$VLM_CPUS"
  --output="$VLM_LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$VLM_PARTITION" ]]; then
  VLM_SBATCH_ARGS+=(--partition="$VLM_PARTITION")
fi
if [[ -n "$build_job_id" ]]; then
  VLM_SBATCH_ARGS+=(--dependency="afterok:$build_job_id")
  export SUCC_SKIP_DATASET_BUILD=1
else
  export SUCC_SKIP_DATASET_BUILD="${SUCC_SKIP_DATASET_BUILD:-1}"
fi

echo
echo "Submitting HF VLM benchmark job:"
echo "  account=$VLM_ACCOUNT"
echo "  time=$VLM_TIME"
echo "  mem=$VLM_MEM"
echo "  cpus=$VLM_CPUS"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  log_dir=$VLM_LOG_DIR"
if [[ -n "$build_job_id" ]]; then
  echo "  dependency=afterok:$build_job_id"
fi
vlm_job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if vlm_output="$(
    sbatch "${VLM_SBATCH_ARGS[@]}" \
      --gpus="$GPU_REQUEST" \
      --wrap="bash '$PROJECT_DIR/scripts/run_hf_vlm_multiproperty_workflow.sh'"
  )"; then
    echo "$vlm_output"
    vlm_job_id="$(echo "$vlm_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    break
  fi
done
if [[ -z "$vlm_job_id" ]]; then
  echo "ERROR: failed to submit VLM job with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi

echo
echo "Pipeline submitted."
if [[ -n "$build_job_id" ]]; then
  echo "  dataset_job=$build_job_id"
fi
echo "  vlm_job=${vlm_job_id:-unknown}"
echo "  final_report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
if [[ "$SUCC_TRAIN_FEATURE_CONNECTOR" == "1" ]]; then
  echo "  connector_report=$CONNECTOR_BENCHMARK_OUTPUT_DIR/benchmark_report.md"
fi
