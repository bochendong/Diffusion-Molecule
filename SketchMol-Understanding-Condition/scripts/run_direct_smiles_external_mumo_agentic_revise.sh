#!/usr/bin/env bash
# Run MuMO source-conditioned agentic revise benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUTPUT_DIR="${SUCC_EXTERNAL_AGENTIC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1}"
SOURCE_FILE="${SUCC_EXTERNAL_AGENTIC_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
ROWS_CSV="${SUCC_EXTERNAL_AGENTIC_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_rows.csv}"
SUMMARY_JSON="${SUCC_EXTERNAL_AGENTIC_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_rows.summary.json}"
TASK_SPEC_JSON="${SUCC_EXTERNAL_AGENTIC_TASK_SPEC_JSON:-$OUTPUT_DIR/external_multiproperty_task_specs.json}"
FEATURES_DIR="${SUCC_EXTERNAL_AGENTIC_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_source_edit_sft_v1/eval_condition_features_hf_vlm}"
DIRECT_MODEL_DIR="${SUCC_EXTERNAL_AGENTIC_DIRECT_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_agentic_proposal}"
BENCHMARK_OUTPUT_DIR="${SUCC_EXTERNAL_AGENTIC_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_agentic_revise}"
DIRECT_PREDICTION_CSV="${SUCC_EXTERNAL_AGENTIC_DIRECT_PREDICTION_CSV:-$OUTPUT_DIR/direct_smiles_proposals.csv}"
AGENTIC_PREDICTION_CSV="${SUCC_EXTERNAL_AGENTIC_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/agentic_revise_predictions.csv}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_AGENTIC_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_AGENTIC_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"

SUITE="${SUCC_EXTERNAL_AGENTIC_SUITE:-mumo}"
TASK_SPLIT="${SUCC_EXTERNAL_AGENTIC_TASK_SPLIT:-all}"
TASKS="${SUCC_EXTERNAL_AGENTIC_TASKS:-}"
INPUT_SPLIT="${SUCC_EXTERNAL_AGENTIC_INPUT_SPLIT:-all}"
MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_AGENTIC_MAX_ROWS_PER_TASK:-200}"
SEED="${SUCC_EXTERNAL_AGENTIC_SEED:-17}"
FORCE_EXPORT="${SUCC_EXTERNAL_AGENTIC_FORCE_EXPORT:-0}"
RUN_FEATURE_EXPORT="${SUCC_EXTERNAL_AGENTIC_RUN_FEATURE_EXPORT:-auto}"
RUN_DIRECT="${SUCC_EXTERNAL_AGENTIC_RUN_DIRECT:-1}"
RESUME_CHECKPOINT="${SUCC_EXTERNAL_AGENTIC_RESUME_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_source_edit_sft_v1/direct_smiles_model_source_edit_sft/direct_smiles_generator.pt}"
CONDITION_MIXING_MODE="${SUCC_EXTERNAL_AGENTIC_CONDITION_MIXING_MODE:-append_source_property_program}"

EVAL_BATCH_SIZE="${SUCC_EXTERNAL_AGENTIC_EVAL_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${SUCC_EXTERNAL_AGENTIC_MAX_NEW_TOKENS:-100}"
TEMPERATURE="${SUCC_EXTERNAL_AGENTIC_TEMPERATURE:-0.70}"
TOP_K="${SUCC_EXTERNAL_AGENTIC_TOP_K:-24}"
TOP_P="${SUCC_EXTERNAL_AGENTIC_TOP_P:-0.90}"
NUM_SAMPLES="${SUCC_EXTERNAL_AGENTIC_NUM_SAMPLES:-20}"
PARALLEL_SAMPLES="${SUCC_EXTERNAL_AGENTIC_PARALLEL_SAMPLES:-4}"
MAX_PARALLEL_SEQUENCES="${SUCC_EXTERNAL_AGENTIC_MAX_PARALLEL_SEQUENCES:-512}"
REPETITION_PENALTY="${SUCC_EXTERNAL_AGENTIC_REPETITION_PENALTY:-1.15}"
NO_REPEAT_NGRAM_SIZE="${SUCC_EXTERNAL_AGENTIC_NO_REPEAT_NGRAM_SIZE:-6}"
MIN_NEW_TOKENS="${SUCC_EXTERNAL_AGENTIC_MIN_NEW_TOKENS:-6}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_AGENTIC_MIN_SOURCE_TANIMOTO:-0.4}"
AGENTIC_STEPS="${SUCC_EXTERNAL_AGENTIC_STEPS:-2}"
AGENTIC_BEAM_SIZE="${SUCC_EXTERNAL_AGENTIC_BEAM_SIZE:-48}"
AGENTIC_MAX_CANDIDATES_PER_ROW="${SUCC_EXTERNAL_AGENTIC_MAX_CANDIDATES_PER_ROW:-256}"
AGENTIC_MAX_CANDIDATES_PER_PARENT="${SUCC_EXTERNAL_AGENTIC_MAX_CANDIDATES_PER_PARENT:-64}"
AGENTIC_EDIT_ACTION_PROFILE="${SUCC_EXTERNAL_AGENTIC_EDIT_ACTION_PROFILE:-basic}"
AGENTIC_SELECTION_MODE="${SUCC_EXTERNAL_AGENTIC_SELECTION_MODE:-score}"
AGENTIC_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION="${SUCC_EXTERNAL_AGENTIC_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION:-1.0}"
AGENTIC_PROPERTY_WEIGHT="${SUCC_EXTERNAL_AGENTIC_PROPERTY_WEIGHT:-100}"
AGENTIC_DISTANCE_WEIGHT="${SUCC_EXTERNAL_AGENTIC_DISTANCE_WEIGHT:-10}"
AGENTIC_SIMILARITY_WEIGHT="${SUCC_EXTERNAL_AGENTIC_SIMILARITY_WEIGHT:-12}"
AGENTIC_SIMILARITY_BONUS="${SUCC_EXTERNAL_AGENTIC_SIMILARITY_BONUS:-25}"
AGENTIC_COPY_PENALTY="${SUCC_EXTERNAL_AGENTIC_COPY_PENALTY:-2}"
DEVICE="${SUCC_DEVICE:-auto}"

HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-auto}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"
POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$DIRECT_MODEL_DIR" "$BENCHMARK_OUTPUT_DIR"

echo "MuMO external agentic revise benchmark"
echo "  python=$PYTHON_BIN"
echo "  source_file=$SOURCE_FILE"
echo "  output_dir=$OUTPUT_DIR"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  features_dir=$FEATURES_DIR"
echo "  condition_mixing_mode=$CONDITION_MIXING_MODE"
echo "  run_direct=$RUN_DIRECT"
echo "  agentic_steps=$AGENTIC_STEPS"
echo "  agentic_beam_size=$AGENTIC_BEAM_SIZE"
echo "  agentic_max_candidates_per_row=$AGENTIC_MAX_CANDIDATES_PER_ROW"
echo "  agentic_max_candidates_per_parent=$AGENTIC_MAX_CANDIDATES_PER_PARENT"
echo "  agentic_edit_action_profile=$AGENTIC_EDIT_ACTION_PROFILE"
echo "  agentic_selection_mode=$AGENTIC_SELECTION_MODE"
echo "  min_source_tanimoto=$MIN_SOURCE_TANIMOTO"

if [[ "$FORCE_EXPORT" == "1" || ! -f "$ROWS_CSV" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
    --source-file "$SOURCE_FILE" \
    --output-csv "$ROWS_CSV" \
    --summary-json "$SUMMARY_JSON" \
    --task-spec-json "$TASK_SPEC_JSON" \
    --suite "$SUITE" \
    --task-split "$TASK_SPLIT" \
    --tasks "$TASKS" \
    --input-split "$INPUT_SPLIT" \
    --max-rows-per-task "$MAX_ROWS_PER_TASK" \
    --seed "$SEED"
fi

should_export_features=0
if [[ "$RUN_FEATURE_EXPORT" == "1" || "$FORCE_EXPORT" == "1" ]]; then
  should_export_features=1
elif [[ "$RUN_FEATURE_EXPORT" == "auto" && ! -f "$FEATURES_DIR/query_tokens.npy" ]]; then
  should_export_features=1
fi
if [[ "$should_export_features" == "1" ]]; then
  export SUCC_ENCODER=hf_vlm
  export SUCC_VARIANTS=full
  export SUCC_BASELINE_CSV="$ROWS_CSV"
  export SUCC_OUTPUT_DIR="$FEATURES_DIR"
  export SUCC_POOLED_DIM="$POOLED_DIM"
  export SUCC_NUM_QUERIES="$NUM_QUERIES"
  export SUCC_QUERY_DIM="$QUERY_DIM"
  export SUCC_HF_MODEL_NAME_OR_PATH="$HF_MODEL_NAME_OR_PATH"
  export SUCC_HF_DEVICE_MAP="$HF_DEVICE_MAP"
  export SUCC_HF_DTYPE="$HF_DTYPE"
  export SUCC_HF_BATCH_SIZE="$HF_BATCH_SIZE"
  export SUCC_HF_MAX_LENGTH="$HF_MAX_LENGTH"
  export SUCC_HF_RENDER_IMAGE_SIZE="$HF_RENDER_IMAGE_SIZE"
  bash "$PROJECT_DIR/scripts/run_condition_encoder_export.sh"
fi

if [[ "$RUN_DIRECT" == "1" || ! -f "$DIRECT_PREDICTION_CSV" ]]; then
  FEATURE_ARGS=()
  if [[ -f "$FEATURES_DIR/query_tokens.npy" ]]; then
    FEATURE_ARGS+=(--condition-features-dir "$FEATURES_DIR" --eval-condition-features-dir "$FEATURES_DIR")
  fi
  DIRECT_CMD=(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py"
    --eval-only
    --eval-csv "$ROWS_CSV"
    --resume-checkpoint "$RESUME_CHECKPOINT"
    "${FEATURE_ARGS[@]}"
    --condition-mixing-mode "$CONDITION_MIXING_MODE"
    --output-dir "$DIRECT_MODEL_DIR"
    --prediction-csv "$DIRECT_PREDICTION_CSV"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --num-samples "$NUM_SAMPLES"
    --parallel-samples "$PARALLEL_SAMPLES"
    --max-parallel-sequences "$MAX_PARALLEL_SEQUENCES"
    --repetition-penalty "$REPETITION_PENALTY"
    --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE"
    --min-new-tokens "$MIN_NEW_TOKENS"
    --disable-property-rerank
    --seed "$SEED"
    --device "$DEVICE"
  )
  "${DIRECT_CMD[@]}"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/build_external_agentic_revise_predictions.py" \
  --rows-csv "$ROWS_CSV" \
  --direct-prediction-csv "$DIRECT_PREDICTION_CSV" \
  --prediction-csv "$AGENTIC_PREDICTION_CSV" \
  --min-source-tanimoto "$MIN_SOURCE_TANIMOTO" \
  --max-steps "$AGENTIC_STEPS" \
  --beam-size "$AGENTIC_BEAM_SIZE" \
  --max-candidates-per-row "$AGENTIC_MAX_CANDIDATES_PER_ROW" \
  --max-candidates-per-parent "$AGENTIC_MAX_CANDIDATES_PER_PARENT" \
  --edit-action-profile "$AGENTIC_EDIT_ACTION_PROFILE" \
  --selection-mode "$AGENTIC_SELECTION_MODE" \
  --similarity-first-min-local-success-fraction "$AGENTIC_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION" \
  --property-weight "$AGENTIC_PROPERTY_WEIGHT" \
  --distance-weight "$AGENTIC_DISTANCE_WEIGHT" \
  --similarity-weight "$AGENTIC_SIMILARITY_WEIGHT" \
  --similarity-bonus "$AGENTIC_SIMILARITY_BONUS" \
  --copy-penalty "$AGENTIC_COPY_PENALTY" \
  --seed "$SEED"

EVAL_ARGS=()
if [[ -n "$GENERATED_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--generated-properties-csv "$GENERATED_PROPERTIES_CSV")
fi
if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
  --prediction-csv "$AGENTIC_PREDICTION_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --smiles-column generated_smiles \
  --source-smiles-column source_smiles \
  --min-source-tanimoto "$MIN_SOURCE_TANIMOTO" \
  --report-title "SUCC External MuMO Agentic Revise Benchmark" \
  "${EVAL_ARGS[@]}"

echo
echo "MuMO agentic revise benchmark ready:"
echo "  rows=$ROWS_CSV"
echo "  direct_predictions=$DIRECT_PREDICTION_CSV"
echo "  agentic_predictions=$AGENTIC_PREDICTION_CSV"
echo "  report=$BENCHMARK_OUTPUT_DIR/external_multiproperty_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/external_multiproperty_summary.csv"
