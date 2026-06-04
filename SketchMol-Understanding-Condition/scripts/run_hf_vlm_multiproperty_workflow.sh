#!/usr/bin/env bash
# Run the big-VLM understanding workflow on the multi-property edit dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
DATASET_PROJECT_DIR="$REPO_DIR/SketchMol-MultiProperty-EditDataset"
cd "$REPO_DIR"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023
module load python/3.11
module load rdkit/2025.09.4

PYTHON_BIN="${SUCC_PYTHON_BIN:-${SMMED_PYTHON_BIN:-${PYTHON_BIN:-python}}}"
export PYTHONPATH="$PROJECT_DIR:$DATASET_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
BASELINE_CSV="${SUCC_BASELINE_CSV:-$DATASET_OUTPUT_DIR/baseline_variants.csv}"
CONDITION_ROWS="${SMMED_CONDITION_ROWS_CSV:-$DATASET_OUTPUT_DIR/condition_rows.csv}"
FEATURES_DIR="${SUCC_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm}"
BENCHMARK_OUTPUT_DIR="${SMMED_BENCHMARK_OUTPUT_DIR:-$DATASET_OUTPUT_DIR/benchmark_hf_vlm}"
HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-bfloat16}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_ATTN_IMPLEMENTATION="${SUCC_HF_ATTN_IMPLEMENTATION:-}"
HF_PROMPT_STYLE="${SUCC_HF_PROMPT_STYLE:-auto}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"
HF_TRUST_REMOTE_CODE="${SUCC_HF_TRUST_REMOTE_CODE:-1}"
POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"
SKIP_DATASET_BUILD="${SUCC_SKIP_DATASET_BUILD:-0}"
SKIP_FEATURE_EXPORT="${SUCC_SKIP_FEATURE_EXPORT:-0}"
METHODS="${SMMED_BENCHMARK_METHODS:-source_identity,vlm_scaffold_feature_retrieval,target_oracle}"
LIMIT_EVAL_ROWS="${SMMED_LIMIT_EVAL_ROWS:-}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"
MAX_FEATURE_CANDIDATES="${SMMED_MAX_FEATURE_CANDIDATES:-20000}"

echo "HF VLM multi-property understanding workflow"
echo "  python=$PYTHON_BIN"
echo "  model=$HF_MODEL_NAME_OR_PATH"
echo "  dataset_output_dir=$DATASET_OUTPUT_DIR"
echo "  baseline_csv=$BASELINE_CSV"
echo "  condition_rows=$CONDITION_ROWS"
echo "  features_dir=$FEATURES_DIR"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  methods=$METHODS"

if [[ "$SKIP_DATASET_BUILD" != "1" ]]; then
  bash "$DATASET_PROJECT_DIR/scripts/run_build_dataset.sh"
fi

if [[ ! -f "$BASELINE_CSV" || ! -f "$CONDITION_ROWS" ]]; then
  echo "ERROR: multi-property dataset files are missing." >&2
  echo "Expected baseline CSV: $BASELINE_CSV" >&2
  echo "Expected condition rows: $CONDITION_ROWS" >&2
  exit 2
fi

if [[ "$SKIP_FEATURE_EXPORT" != "1" ]]; then
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ENCODER=hf_vlm \
  SUCC_VARIANTS=full \
  SUCC_BASELINE_CSV="$BASELINE_CSV" \
  SUCC_OUTPUT_DIR="$FEATURES_DIR" \
  SUCC_HF_MODEL_NAME_OR_PATH="$HF_MODEL_NAME_OR_PATH" \
  SUCC_HF_DEVICE_MAP="$HF_DEVICE_MAP" \
  SUCC_HF_DTYPE="$HF_DTYPE" \
  SUCC_HF_BATCH_SIZE="$HF_BATCH_SIZE" \
  SUCC_HF_MAX_LENGTH="$HF_MAX_LENGTH" \
  SUCC_HF_ATTN_IMPLEMENTATION="$HF_ATTN_IMPLEMENTATION" \
  SUCC_HF_PROMPT_STYLE="$HF_PROMPT_STYLE" \
  SUCC_HF_RENDER_IMAGE_SIZE="$HF_RENDER_IMAGE_SIZE" \
  SUCC_HF_TRUST_REMOTE_CODE="$HF_TRUST_REMOTE_CODE" \
  SUCC_POOLED_DIM="$POOLED_DIM" \
  SUCC_NUM_QUERIES="$NUM_QUERIES" \
  SUCC_QUERY_DIM="$QUERY_DIM" \
  bash "$PROJECT_DIR/scripts/run_condition_encoder_export.sh"
fi

LIMIT_EVAL_ARGS=()
if [[ -n "$LIMIT_EVAL_ROWS" ]]; then
  LIMIT_EVAL_ARGS=(--limit-eval-rows "$LIMIT_EVAL_ROWS")
fi

"$PYTHON_BIN" "$DATASET_PROJECT_DIR/scripts/benchmark_multiproperty_retrieval.py" \
  --condition-rows-csv "$CONDITION_ROWS" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --methods "$METHODS" \
  --condition-features-dir "$FEATURES_DIR" \
  --condition-feature-array pooled \
  --condition-feature-variant full \
  --max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT" \
  --max-feature-candidates "$MAX_FEATURE_CANDIDATES" \
  "${LIMIT_EVAL_ARGS[@]}"

echo
echo "HF VLM multi-property benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
echo
sed -n '1,80p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
