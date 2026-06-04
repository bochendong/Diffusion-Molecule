#!/usr/bin/env bash
# Submit the full contrastive-fusion understanding benchmark workflow to Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

ACCOUNT="${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"

BASELINE_CSV="${SUCC_BASELINE_CSV:-SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv}"
VARIANT="${SUCC_VARIANT:-full}"
RUN_TAG="${SUCC_RUN_TAG:-fusion_v2_contrastive_e12}"

FUSION_OUTPUT_DIR="${SUCC_FUSION_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/fusion_image_text_encoder_mixed_v2_contrastive_e12}"
FEATURES_DIR="${SUCC_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_multimodal_fusion_v2_contrastive_e12}"
DELTA_OUTPUT_DIR="${SUCC_DELTA_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/delta_bucket_classifier_multimodal_fusion_v2_contrastive_e12}"
BENCHMARK_EXPORT_DIR="${SUCC_BENCHMARK_EXPORT_DIR:-SketchMol-Understanding-Condition/outputs/benchmark_export_fusion_v2_contrastive_e12}"
BENCHMARK_OUTPUT_DIR="${SUCC_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/understanding_condition_fusion_v2_contrastive_e12}"
COMPARE_OUTPUT_DIR="${SUCC_COMPARE_OUTPUT_DIR:-SketchMolCompare/outputs/comparisons/understanding_contrastive_e12}"

TRAIN_TIME="${SUCC_TRAIN_TIME:-04:00:00}"
TRAIN_MEM="${SUCC_TRAIN_MEM:-16G}"
TRAIN_CPUS="${SUCC_TRAIN_CPUS:-4}"
EXPORT_TIME="${SUCC_EXPORT_TIME:-00:45:00}"
EXPORT_MEM="${SUCC_EXPORT_MEM:-16G}"
EXPORT_CPUS="${SUCC_EXPORT_CPUS:-2}"
PROBE_TIME="${SUCC_PROBE_TIME:-00:30:00}"
PROBE_MEM="${SUCC_PROBE_MEM:-8G}"
PROBE_CPUS="${SUCC_PROBE_CPUS:-1}"
BENCHMARK_TIME="${SUCC_BENCHMARK_TIME:-00:45:00}"
BENCHMARK_MEM="${SUCC_BENCHMARK_MEM:-8G}"
BENCHMARK_CPUS="${SUCC_BENCHMARK_CPUS:-1}"
COMPARE_TIME="${SUCC_COMPARE_TIME:-00:15:00}"
COMPARE_MEM="${SUCC_COMPARE_MEM:-4G}"
COMPARE_CPUS="${SUCC_COMPARE_CPUS:-1}"

mkdir -p "$LOG_DIR"

submit_job() {
  local name="$1"
  local time_limit="$2"
  local mem="$3"
  local cpus="$4"
  local dependency="$5"
  local command="$6"

  local args=(
    --parsable
    --account="$ACCOUNT"
    --job-name="$name"
    --time="$time_limit"
    --mem="$mem"
    --cpus-per-task="$cpus"
    --output="$LOG_DIR/%x-%j.log"
    --export=ALL
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="$dependency")
  fi
  sbatch "${args[@]}" --wrap="$command"
}

echo "Submitting contrastive understanding benchmark workflow"
echo "  account=$ACCOUNT"
echo "  python=$PYTHON_BIN"
echo "  baseline_csv=$BASELINE_CSV"
echo "  run_tag=$RUN_TAG"
echo "  logs=$LOG_DIR"
echo

TRAIN_JOB="$(
  submit_job \
    "succ-train-$RUN_TAG" \
    "$TRAIN_TIME" \
    "$TRAIN_MEM" \
    "$TRAIN_CPUS" \
    "" \
    "SUCC_PYTHON_BIN='$PYTHON_BIN' SUCC_BASELINE_CSV='$BASELINE_CSV' SUCC_OUTPUT_DIR='$FUSION_OUTPUT_DIR' bash '$PROJECT_DIR/scripts/run_contrastive_fusion_encoder.sh'"
)"
echo "train_job=$TRAIN_JOB"

EXPORT_JOB="$(
  submit_job \
    "succ-export-$RUN_TAG" \
    "$EXPORT_TIME" \
    "$EXPORT_MEM" \
    "$EXPORT_CPUS" \
    "afterok:$TRAIN_JOB" \
    "SUCC_PYTHON_BIN='$PYTHON_BIN' SUCC_BASELINE_CSV='$BASELINE_CSV' SUCC_ENCODER=multimodal_fusion_v2 SUCC_IMAGE_ENCODER_CHECKPOINT='$FUSION_OUTPUT_DIR/fusion_image_text_encoder.pt' SUCC_OUTPUT_DIR='$FEATURES_DIR' bash '$PROJECT_DIR/scripts/run_condition_encoder_export.sh'"
)"
echo "export_job=$EXPORT_JOB"

PROBE_JOB="$(
  submit_job \
    "succ-probe-$RUN_TAG" \
    "$PROBE_TIME" \
    "$PROBE_MEM" \
    "$PROBE_CPUS" \
    "afterok:$EXPORT_JOB" \
    "SUCC_BASELINE_CSV='$BASELINE_CSV' SUCC_CONDITION_FEATURES_DIR='$FEATURES_DIR' SUCC_OUTPUT_DIR='$DELTA_OUTPUT_DIR' bash '$PROJECT_DIR/scripts/run_delta_bucket_classifier.sh'"
)"
echo "probe_job=$PROBE_JOB"

BENCHMARK_JOB="$(
  submit_job \
    "succ-bench-$RUN_TAG" \
    "$BENCHMARK_TIME" \
    "$BENCHMARK_MEM" \
    "$BENCHMARK_CPUS" \
    "afterok:$EXPORT_JOB" \
    "SUCC_PYTHON_BIN='$PYTHON_BIN' SUCC_BASELINE_VARIANTS_CSV='$BASELINE_CSV' SUCC_VARIANT='$VARIANT' SUCC_CONDITION_FEATURES_DIR='$FEATURES_DIR' SUCC_BENCHMARK_EXPORT_DIR='$BENCHMARK_EXPORT_DIR' SUCC_BENCHMARK_OUTPUT_DIR='$BENCHMARK_OUTPUT_DIR' bash '$PROJECT_DIR/scripts/run_benchmark_export.sh'"
)"
echo "benchmark_job=$BENCHMARK_JOB"

COMPARE_SUMMARIES="SketchMolBenchmark/outputs/current/benchmark_summary.csv SketchMolBenchmark/outputs/direct_structure_current/benchmark_summary.csv SketchMolBenchmark/outputs/understanding_condition_full/benchmark_summary.csv $BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
COMPARE_JOB="$(
  submit_job \
    "succ-compare-$RUN_TAG" \
    "$COMPARE_TIME" \
    "$COMPARE_MEM" \
    "$COMPARE_CPUS" \
    "afterok:$BENCHMARK_JOB" \
    "SKETCHMOL_COMPARE_SKETCHMOL_SUMMARIES='$COMPARE_SUMMARIES' SKETCHMOL_COMPARE_OUT='$COMPARE_OUTPUT_DIR' bash '$REPO_DIR/SketchMolCompare/scripts/run_compare_existing.sh'"
)"
echo "compare_job=$COMPARE_JOB"

echo
echo "Workflow submitted."
echo "  train_checkpoint=$FUSION_OUTPUT_DIR/fusion_image_text_encoder.pt"
echo "  condition_features=$FEATURES_DIR"
echo "  delta_metrics=$DELTA_OUTPUT_DIR/metrics.csv"
echo "  benchmark_summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  compare_report=$COMPARE_OUTPUT_DIR/comparison_report.md"
