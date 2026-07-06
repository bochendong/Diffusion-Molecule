#!/usr/bin/env bash
# Run official GeLLMO-C inference for one C-MuMO task and convert responses.

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

PYTHON_BIN="${SUCC_GELLMOC_PYTHON_BIN:-${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}}"
CONVERT_PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
GELLMOC_REPO_URL="${SUCC_GELLMOC_REPO_URL:-https://github.com/ninglab/GeLLMO-C.git}"
GELLMOC_REPO_DIR="${SUCC_GELLMOC_REPO_DIR:-Research/Molecule Generation/OfficialBaselines/GeLLMO-C/repo}"
GELLMOC_ARCHIVE_URL="${SUCC_GELLMOC_ARCHIVE_URL:-https://github.com/ninglab/GeLLMO-C/archive/refs/heads/main.tar.gz}"
SOURCE_FILE="${SUCC_GELLMOC_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/cmumo/test.json}"
DATA_PATH="${SUCC_GELLMOC_DATA_PATH:-NingLab/C-MuMOInstruct}"
BASE_MODEL="${SUCC_GELLMOC_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
LORA_WEIGHTS="${SUCC_GELLMOC_LORA_WEIGHTS:-NingLab/GeLLMO-C-P6-Mistral}"
MODEL_ID="${SUCC_GELLMOC_MODEL_ID:-mistral}"
TASK="${SUCC_GELLMOC_TASK:-BPQ}"
SETTING="${SUCC_GELLMOC_SETTING:-seen}"
NUM_SEQ="${SUCC_GELLMOC_NUM_SEQ:-20}"
MAX_ROWS="${SUCC_GELLMOC_MAX_ROWS:-0}"
OUTPUT_DIR="${SUCC_GELLMOC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_gellmoc_official_cmumo_v1}"
TASK_SLUG="${TASK//+/_}"
TASK_OUTPUT_DIR="${SUCC_GELLMOC_TASK_OUTPUT_DIR:-$OUTPUT_DIR/tasks/${TASK_SLUG}_${SETTING}}"
RAW_OUTPUT_DIR="$TASK_OUTPUT_DIR/raw"
RESPONSE_JSON="${SUCC_GELLMOC_RESPONSE_JSON:-}"
PREDICTION_CSV="${SUCC_GELLMOC_PREDICTION_CSV:-$TASK_OUTPUT_DIR/gellmoc_candidate_predictions.csv}"
GENERATED_PROPERTIES_CSV="${SUCC_GELLMOC_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_GELLMOC_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"
EVAL_OUTPUT_DIR="${SUCC_GELLMOC_EVAL_OUTPUT_DIR:-$TASK_OUTPUT_DIR/eval_with_existing_oracle}"
RUN_EVAL="${SUCC_GELLMOC_RUN_EVAL:-0}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

echo "Official GeLLMO-C task inference"
echo "  official_repo=$GELLMOC_REPO_DIR"
echo "  task=$TASK"
echo "  setting=$SETTING"
echo "  num_seq=$NUM_SEQ"
echo "  base_model=$BASE_MODEL"
echo "  lora_weights=$LORA_WEIGHTS"
echo "  data_path=$DATA_PATH"
echo "  source_file=$SOURCE_FILE"
echo "  output_dir=$TASK_OUTPUT_DIR"
echo "  python=$PYTHON_BIN"

ensure_gellmoc_repo() {
  if [[ -f "$GELLMOC_REPO_DIR/inference.sh" || -f "$GELLMOC_REPO_DIR/inference.py" ]]; then
    echo "exists: $GELLMOC_REPO_DIR"
    return
  fi
  if [[ -e "$GELLMOC_REPO_DIR" ]]; then
    echo "ERROR: GeLLMO-C repo path exists but has no inference.sh/inference.py: $GELLMOC_REPO_DIR" >&2
    echo "Run: bash SketchMol-Understanding-Condition/scripts/sync_official_code.sh --method cmumo" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$GELLMOC_REPO_DIR")"
  if git clone --depth 1 "$GELLMOC_REPO_URL" "$GELLMOC_REPO_DIR"; then
    return
  fi
  echo "WARN: git clone failed; falling back to GitHub tarball." >&2
  mkdir -p "$GELLMOC_REPO_DIR"
  curl -L --fail "$GELLMOC_ARCHIVE_URL" | tar -xz --strip-components=1 -C "$GELLMOC_REPO_DIR"
}

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "ERROR: missing local C-MuMO source file for conversion: $SOURCE_FILE" >&2
  echo "Run: bash SketchMol-Understanding-Condition/scripts/fetch_external_cmumo_dataset.sh" >&2
  exit 2
fi

ensure_gellmoc_repo
mkdir -p "$RAW_OUTPUT_DIR" "$TASK_OUTPUT_DIR"

(
  cd "$GELLMOC_REPO_DIR"
  if [[ -f inference.sh ]]; then
    bash inference.sh \
      "$BASE_MODEL" \
      "$DATA_PATH" \
      "$LORA_WEIGHTS" \
      "$RAW_OUTPUT_DIR" \
      "$TASK" \
      "$NUM_SEQ" \
      "$SETTING"
  elif [[ -f inference.py ]]; then
    "$PYTHON_BIN" inference.py \
      --data_path "$DATA_PATH" \
      --output_dir "$RAW_OUTPUT_DIR" \
      --lora_weights "$LORA_WEIGHTS" \
      --base_model "$BASE_MODEL" \
      --setting "$SETTING" \
      --opt_type simple \
      --task "$TASK" \
      --num_return_sequences "$NUM_SEQ"
  else
    echo "ERROR: GeLLMO-C repo has neither inference.sh nor inference.py." >&2
    exit 2
  fi
)

if [[ -z "$RESPONSE_JSON" ]]; then
  response_candidate="$(find "$RAW_OUTPUT_DIR" -type f -name "*response.json" -print -quit)"
  RESPONSE_JSON="$response_candidate"
fi
if [[ -z "$RESPONSE_JSON" || ! -f "$RESPONSE_JSON" ]]; then
  echo "ERROR: official GeLLMO-C response JSON not found under: $RAW_OUTPUT_DIR" >&2
  exit 1
fi

"$CONVERT_PYTHON_BIN" "$PROJECT_DIR/scripts/convert_external_gellmo_responses.py" \
  --response-json "$RESPONSE_JSON" \
  --source-file "$SOURCE_FILE" \
  --output-csv "$PREDICTION_CSV" \
  --suite cmumo \
  --task "$TASK" \
  --setting "$SETTING" \
  --max-rows "$MAX_ROWS" \
  --model-id "$MODEL_ID" \
  --base-model "$BASE_MODEL" \
  --lora-weights "$LORA_WEIGHTS" \
  --method gellmoc_official

RUN_MANIFEST="$TASK_OUTPUT_DIR/official_run_manifest.json"
repo_commit="$(git -C "$GELLMOC_REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
"$CONVERT_PYTHON_BIN" - <<'PY' "$RUN_MANIFEST" "$GELLMOC_REPO_URL" "$GELLMOC_REPO_DIR" "$repo_commit" "$DATA_PATH" "$BASE_MODEL" "$LORA_WEIGHTS" "$TASK" "$SETTING" "$NUM_SEQ" "$RESPONSE_JSON" "$PREDICTION_CSV"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "benchmark_id": "cmumo",
    "method_id": "GeLLMO-C",
    "reproduction_type": "official_code",
    "official_repo_url": sys.argv[2],
    "official_repo_dir": sys.argv[3],
    "official_repo_commit": sys.argv[4] or None,
    "data_path": sys.argv[5],
    "base_model": sys.argv[6],
    "lora_weights": sys.argv[7],
    "task": sys.argv[8],
    "setting": sys.argv[9],
    "num_return_sequences": int(sys.argv[10]),
    "response_json": sys.argv[11],
    "prediction_csv": sys.argv[12],
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ -z "$GENERATED_PROPERTIES_CSV" ]]; then
    echo "ERROR: SUCC_GELLMOC_RUN_EVAL=1 requires SUCC_GELLMOC_GENERATED_PROPERTIES_CSV." >&2
    exit 2
  fi
  eval_args=(
    --prediction-csv "$PREDICTION_CSV"
    --output-dir "$EVAL_OUTPUT_DIR"
    --generated-properties-csv "$GENERATED_PROPERTIES_CSV"
    --smiles-column generated_smiles
    --source-smiles-column source_smiles
    --group-column condition_id
    --report-title "Official GeLLMO-C C-MuMO Baseline"
  )
  if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
    eval_args+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
  fi
  "$CONVERT_PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" "${eval_args[@]}"
fi

echo
echo "Official GeLLMO-C task ready:"
echo "  response_json=$RESPONSE_JSON"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  run_manifest=$RUN_MANIFEST"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "  eval_report=$EVAL_OUTPUT_DIR/external_multiproperty_report.md"
fi
