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

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from rdkit import Chem
import PIL
import torch
import transformers
PY
then
  echo "ERROR: PYTHON_BIN=$PYTHON_BIN cannot import the required VLM workflow packages." >&2
  echo "       It must provide torch, transformers, PIL, and RDKit." >&2
  echo "       On Nibi, use a unified VLM/RDKit venv such as /home/bdong/.venvs/molscribe_overlay/bin/python," >&2
  echo "       or set SUCC_PYTHON_BIN to another Python that can import all four packages." >&2
  exit 2
fi

DATASET_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
BASELINE_CSV="${SUCC_BASELINE_CSV:-$DATASET_OUTPUT_DIR/baseline_variants.csv}"
CONDITION_ROWS="${SMMED_CONDITION_ROWS_CSV:-$DATASET_OUTPUT_DIR/condition_rows.csv}"
MOLECULE_DB="${SMMED_MOLECULE_DB_CSV:-$DATASET_OUTPUT_DIR/molecule_database.csv}"
FEATURES_DIR="${SUCC_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm}"
BENCHMARK_OUTPUT_DIR="${SMMED_BENCHMARK_OUTPUT_DIR:-$DATASET_OUTPUT_DIR/benchmark_hf_vlm}"
CONNECTOR_FEATURES_DIR="${SUCC_CONNECTOR_FEATURES_DIR:-${FEATURES_DIR}_edit_connector}"
CONNECTOR_BENCHMARK_OUTPUT_DIR="${SMMED_CONNECTOR_BENCHMARK_OUTPUT_DIR:-${BENCHMARK_OUTPUT_DIR}_edit_connector}"
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
TRAIN_FEATURE_CONNECTOR="${SUCC_TRAIN_FEATURE_CONNECTOR:-1}"
CONNECTOR_EPOCHS="${SUCC_CONNECTOR_EPOCHS:-5}"
CONNECTOR_BATCH_SIZE="${SUCC_CONNECTOR_BATCH_SIZE:-1024}"
CONNECTOR_LEARNING_RATE="${SUCC_CONNECTOR_LEARNING_RATE:-1e-3}"
CONNECTOR_HIDDEN_DIM="${SUCC_CONNECTOR_HIDDEN_DIM:-512}"
CONNECTOR_TRAIN_LIMIT="${SUCC_CONNECTOR_TRAIN_LIMIT:-50000}"
CONNECTOR_SOURCE_FEATURE_WEIGHT="${SUCC_CONNECTOR_SOURCE_FEATURE_WEIGHT:-0.25}"
CONNECTOR_SOURCE_FINGERPRINT_BITS="${SUCC_CONNECTOR_SOURCE_FINGERPRINT_BITS:-256}"
METHODS="${SMMED_FROZEN_BENCHMARK_METHODS:-${SMMED_BENCHMARK_METHODS:-source_identity,global_property_retrieval,scaffold_property_retrieval,vlm_feature_retrieval,vlm_scaffold_feature_retrieval,global_property_vlm_rerank,scaffold_property_vlm_rerank,target_oracle}}"
EDIT_METHODS="${SMMED_EDIT_BENCHMARK_METHODS:-source_identity,global_property_retrieval,scaffold_property_retrieval,edit_latent_global_retrieval,edit_latent_scaffold_retrieval,edit_latent_scaffold_source_rerank,target_oracle}"
LIMIT_EVAL_ROWS="${SMMED_LIMIT_EVAL_ROWS:-}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMMED_MAX_EVAL_PER_PROPERTY_COUNT:-5000}"
MAX_FEATURE_CANDIDATES="${SMMED_MAX_FEATURE_CANDIDATES:-20000}"
MAX_GLOBAL_CANDIDATES="${SMMED_MAX_GLOBAL_CANDIDATES:-20000}"
MAX_EDIT_LATENT_CANDIDATES="${SMMED_MAX_EDIT_LATENT_CANDIDATES:-20000}"
RERANK_CANDIDATES="${SMMED_RERANK_CANDIDATES:-64}"
RERANK_PROPERTY_WEIGHT="${SMMED_RERANK_PROPERTY_WEIGHT:-0.5}"
EDIT_LATENT_PROPERTY_WEIGHT="${SMMED_EDIT_LATENT_PROPERTY_WEIGHT:-1.0}"
EDIT_LATENT_DELTA_WEIGHT="${SMMED_EDIT_LATENT_DELTA_WEIGHT:-0.35}"
EDIT_LATENT_DIRECTION_WEIGHT="${SMMED_EDIT_LATENT_DIRECTION_WEIGHT:-0.10}"
EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT="${SMMED_EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT:-0.25}"
SCAFFOLD_FALLBACK_MODE="${SMMED_SCAFFOLD_FALLBACK_MODE:-source_identity}"

echo "HF VLM multi-property understanding workflow"
echo "  python=$PYTHON_BIN"
echo "  model=$HF_MODEL_NAME_OR_PATH"
echo "  dataset_output_dir=$DATASET_OUTPUT_DIR"
echo "  baseline_csv=$BASELINE_CSV"
echo "  condition_rows=$CONDITION_ROWS"
echo "  molecule_db=$MOLECULE_DB"
echo "  features_dir=$FEATURES_DIR"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  train_feature_connector=$TRAIN_FEATURE_CONNECTOR"
echo "  connector_features_dir=$CONNECTOR_FEATURES_DIR"
echo "  connector_benchmark_output_dir=$CONNECTOR_BENCHMARK_OUTPUT_DIR"
echo "  frozen_methods=$METHODS"
echo "  edit_methods=$EDIT_METHODS"
echo "  scaffold_fallback_mode=$SCAFFOLD_FALLBACK_MODE"
echo "  rerank_candidates=$RERANK_CANDIDATES"
echo "  rerank_property_weight=$RERANK_PROPERTY_WEIGHT"
echo "  edit_latent_source_similarity_weight=$EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT"

if [[ "$SKIP_DATASET_BUILD" != "1" ]]; then
  bash "$DATASET_PROJECT_DIR/scripts/run_build_dataset.sh"
fi

if [[ ! -f "$BASELINE_CSV" || ! -f "$CONDITION_ROWS" ]]; then
  echo "ERROR: multi-property dataset files are missing." >&2
  echo "Expected baseline CSV: $BASELINE_CSV" >&2
  echo "Expected condition rows: $CONDITION_ROWS" >&2
  exit 2
fi
if [[ ! -f "$MOLECULE_DB" ]]; then
  echo "ERROR: molecule database is missing." >&2
  echo "Expected molecule DB: $MOLECULE_DB" >&2
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

run_benchmark() {
  local features_dir="$1"
  local output_dir="$2"
  local label="$3"
  local methods="$4"
  local edit_latent_dir="${5:-}"
  EDIT_LATENT_ARGS=()
  if [[ -n "$edit_latent_dir" ]]; then
    EDIT_LATENT_ARGS=(
      --edit-latent-dir "$edit_latent_dir"
      --max-edit-latent-candidates "$MAX_EDIT_LATENT_CANDIDATES"
      --edit-latent-property-weight "$EDIT_LATENT_PROPERTY_WEIGHT"
      --edit-latent-delta-weight "$EDIT_LATENT_DELTA_WEIGHT"
      --edit-latent-direction-weight "$EDIT_LATENT_DIRECTION_WEIGHT"
      --edit-latent-source-similarity-weight "$EDIT_LATENT_SOURCE_SIMILARITY_WEIGHT"
    )
  fi
  echo
  echo "Running benchmark: $label"
  echo "  features=$features_dir"
  echo "  output=$output_dir"
  echo "  methods=$methods"
  if [[ -n "$edit_latent_dir" ]]; then
    echo "  edit_latent=$edit_latent_dir"
  fi
  "$PYTHON_BIN" "$DATASET_PROJECT_DIR/scripts/benchmark_multiproperty_retrieval.py" \
    --condition-rows-csv "$CONDITION_ROWS" \
    --output-dir "$output_dir" \
    --candidate-molecule-db-csv "$MOLECULE_DB" \
    --methods "$methods" \
    --condition-features-dir "$features_dir" \
    --condition-feature-array pooled \
    --condition-feature-variant full \
    --max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT" \
    --max-global-candidates "$MAX_GLOBAL_CANDIDATES" \
    --max-feature-candidates "$MAX_FEATURE_CANDIDATES" \
    --rerank-candidates "$RERANK_CANDIDATES" \
    --rerank-property-weight "$RERANK_PROPERTY_WEIGHT" \
    --scaffold-fallback-mode "$SCAFFOLD_FALLBACK_MODE" \
    "${EDIT_LATENT_ARGS[@]}" \
    "${LIMIT_EVAL_ARGS[@]}"
}

run_benchmark "$FEATURES_DIR" "$BENCHMARK_OUTPUT_DIR" "frozen_hf_vlm" "$METHODS"

if [[ "$TRAIN_FEATURE_CONNECTOR" == "1" ]]; then
  CONNECTOR_ARGS=()
  if [[ -n "$CONNECTOR_TRAIN_LIMIT" ]]; then
    CONNECTOR_ARGS+=(--train-limit "$CONNECTOR_TRAIN_LIMIT")
  fi
  echo
  echo "Training VLM edit connector"
  echo "  output=$CONNECTOR_FEATURES_DIR"
  echo "  epochs=$CONNECTOR_EPOCHS"
  echo "  batch_size=$CONNECTOR_BATCH_SIZE"
  echo "  source_feature_weight=$CONNECTOR_SOURCE_FEATURE_WEIGHT"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_vlm_feature_connector.py" \
    --condition-features-dir "$FEATURES_DIR" \
    --condition-rows-csv "$CONDITION_ROWS" \
    --output-dir "$CONNECTOR_FEATURES_DIR" \
    --epochs "$CONNECTOR_EPOCHS" \
    --batch-size "$CONNECTOR_BATCH_SIZE" \
    --learning-rate "$CONNECTOR_LEARNING_RATE" \
    --hidden-dim "$CONNECTOR_HIDDEN_DIM" \
    --source-feature-weight "$CONNECTOR_SOURCE_FEATURE_WEIGHT" \
    --source-fingerprint-bits "$CONNECTOR_SOURCE_FINGERPRINT_BITS" \
    "${CONNECTOR_ARGS[@]}"
  run_benchmark "$CONNECTOR_FEATURES_DIR" "$CONNECTOR_BENCHMARK_OUTPUT_DIR" "edit_connector" "$EDIT_METHODS" "$CONNECTOR_FEATURES_DIR"
fi

echo
echo "HF VLM multi-property benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  decoded=$BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
if [[ "$TRAIN_FEATURE_CONNECTOR" == "1" ]]; then
  echo "  connector_report=$CONNECTOR_BENCHMARK_OUTPUT_DIR/benchmark_report.md"
  echo "  connector_summary=$CONNECTOR_BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
  echo "  connector_decoded=$CONNECTOR_BENCHMARK_OUTPUT_DIR/benchmark_decoded.csv"
fi
echo
sed -n '1,80p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
if [[ "$TRAIN_FEATURE_CONNECTOR" == "1" ]]; then
  echo
  sed -n '1,80p' "$CONNECTOR_BENCHMARK_OUTPUT_DIR/benchmark_report.md"
fi
