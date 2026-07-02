#!/usr/bin/env bash
# Run official GeLLMO inference for one MuMO task and convert responses to SUCC CSV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4 || true
fi

PYTHON_BIN="${SUCC_GELLMO_PYTHON_BIN:-${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}}"
CONVERT_PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
GELLMO_REPO_URL="${SUCC_GELLMO_REPO_URL:-https://github.com/ninglab/GeLLMO.git}"
GELLMO_REPO_DIR="${SUCC_GELLMO_REPO_DIR:-SketchMol-Understanding-Condition/outputs/external_gellmo_official_repo_v1/GeLLMO}"
GELLMO_ARCHIVE_URL="${SUCC_GELLMO_ARCHIVE_URL:-https://github.com/ninglab/GeLLMO/archive/refs/heads/main.tar.gz}"
SOURCE_FILE="${SUCC_GELLMO_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
DATA_PATH="${SUCC_GELLMO_DATA_PATH:-NingLab/MuMOInstruct}"
BASE_MODEL="${SUCC_GELLMO_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
LORA_WEIGHTS="${SUCC_GELLMO_LORA_WEIGHTS:-NingLab/GeLLMO-P6-Mistral}"
MODEL_ID="${SUCC_GELLMO_MODEL_ID:-mistral}"
TASK="${SUCC_GELLMO_TASK:-bbbp+drd2+qed}"
SETTING="${SUCC_GELLMO_SETTING:-seen}"
NUM_SEQ="${SUCC_GELLMO_NUM_SEQ:-20}"
MAX_ROWS="${SUCC_GELLMO_MAX_ROWS:-0}"
OUTPUT_DIR="${SUCC_GELLMO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_gellmo_official_mumo_v1}"
TASK_SLUG="${TASK//+/_}"
TASK_OUTPUT_DIR="${SUCC_GELLMO_TASK_OUTPUT_DIR:-$OUTPUT_DIR/tasks/${TASK_SLUG}_${SETTING}}"
RAW_OUTPUT_DIR="$TASK_OUTPUT_DIR/raw"
RESPONSE_JSON="${SUCC_GELLMO_RESPONSE_JSON:-$RAW_OUTPUT_DIR/${TASK}_response.json}"
PREDICTION_CSV="${SUCC_GELLMO_PREDICTION_CSV:-$TASK_OUTPUT_DIR/gellmo_candidate_predictions.csv}"
GENERATED_PROPERTIES_CSV="${SUCC_GELLMO_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_GELLMO_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"
EVAL_OUTPUT_DIR="${SUCC_GELLMO_EVAL_OUTPUT_DIR:-$TASK_OUTPUT_DIR/eval_with_existing_oracle}"
RUN_EVAL="${SUCC_GELLMO_RUN_EVAL:-0}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

echo "Official GeLLMO task inference"
echo "  official_repo=$GELLMO_REPO_DIR"
echo "  task=$TASK"
echo "  setting=$SETTING"
echo "  num_seq=$NUM_SEQ"
echo "  base_model=$BASE_MODEL"
echo "  lora_weights=$LORA_WEIGHTS"
echo "  data_path=$DATA_PATH"
echo "  source_file=$SOURCE_FILE"
echo "  output_dir=$TASK_OUTPUT_DIR"
echo "  python=$PYTHON_BIN"

ensure_gellmo_repo() {
  if [[ -f "$GELLMO_REPO_DIR/inference.py" ]]; then
    echo "exists: $GELLMO_REPO_DIR"
    return
  fi
  mkdir -p "$(dirname "$GELLMO_REPO_DIR")"
  rm -rf "$GELLMO_REPO_DIR"
  if git clone --depth 1 "$GELLMO_REPO_URL" "$GELLMO_REPO_DIR"; then
    return
  fi
  echo "WARN: git clone failed; falling back to GitHub tarball." >&2
  mkdir -p "$GELLMO_REPO_DIR"
  curl -L --fail "$GELLMO_ARCHIVE_URL" | tar -xz --strip-components=1 -C "$GELLMO_REPO_DIR"
}

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "ERROR: missing local MuMO source file for conversion: $SOURCE_FILE" >&2
  echo "Run: bash SketchMol-Understanding-Condition/scripts/fetch_external_mumo_dataset.sh" >&2
  exit 2
fi

ensure_gellmo_repo
mkdir -p "$RAW_OUTPUT_DIR" "$TASK_OUTPUT_DIR"

(
  cd "$GELLMO_REPO_DIR"
  "$PYTHON_BIN" inference.py \
    --data_path "$DATA_PATH" \
    --output_dir "$RAW_OUTPUT_DIR" \
    --lora_weights "$LORA_WEIGHTS" \
    --base_model "$BASE_MODEL" \
    --setting "$SETTING" \
    --opt_type simple \
    --task "$TASK" \
    --num_return_sequences "$NUM_SEQ"
)

if [[ ! -f "$RESPONSE_JSON" ]]; then
  echo "ERROR: official response JSON not found: $RESPONSE_JSON" >&2
  exit 1
fi

"$CONVERT_PYTHON_BIN" "$PROJECT_DIR/scripts/convert_external_gellmo_responses.py" \
  --response-json "$RESPONSE_JSON" \
  --source-file "$SOURCE_FILE" \
  --output-csv "$PREDICTION_CSV" \
  --suite mumo \
  --task "$TASK" \
  --setting "$SETTING" \
  --max-rows "$MAX_ROWS" \
  --model-id "$MODEL_ID" \
  --base-model "$BASE_MODEL" \
  --lora-weights "$LORA_WEIGHTS"

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ -z "$GENERATED_PROPERTIES_CSV" ]]; then
    echo "ERROR: SUCC_GELLMO_RUN_EVAL=1 requires SUCC_GELLMO_GENERATED_PROPERTIES_CSV." >&2
    exit 2
  fi
  eval_args=(
    --prediction-csv "$PREDICTION_CSV"
    --output-dir "$EVAL_OUTPUT_DIR"
    --generated-properties-csv "$GENERATED_PROPERTIES_CSV"
    --smiles-column generated_smiles
    --source-smiles-column source_smiles
    --group-column condition_id
    --report-title "Official GeLLMO MuMO Baseline"
  )
  if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
    eval_args+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
  fi
  "$CONVERT_PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" "${eval_args[@]}"
fi

echo
echo "Official GeLLMO task ready:"
echo "  response_json=$RESPONSE_JSON"
echo "  prediction_csv=$PREDICTION_CSV"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "  eval_report=$EVAL_OUTPUT_DIR/external_multiproperty_report.md"
fi
