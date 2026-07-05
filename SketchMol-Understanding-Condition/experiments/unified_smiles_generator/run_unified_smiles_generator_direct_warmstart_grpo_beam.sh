#!/usr/bin/env bash
# Direct-checkpoint warm-start for unified de novo, then GRPO + beam@40 eval.
#
# This is the P0 rescue path for the unified generator line: keep the strong
# direct-SMILES property-program condition layout, train only de novo rows, and
# evaluate with beam decoding before re-introducing edit/external rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SOURCE_SUITE_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
OUTPUT_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_direct_warmstart_grpo_beam_v1}"
DATA_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_DATA_ROOT:-$OUTPUT_ROOT/dataset}"
SOURCE_TRAIN_CSV="${SUCC_UNIFIED_DIRECT_WARMSTART_SOURCE_TRAIN_CSV:-$SOURCE_SUITE_ROOT/dataset/unified_train_rows.csv}"
SOURCE_EVAL_CSV="${SUCC_UNIFIED_DIRECT_WARMSTART_SOURCE_EVAL_CSV:-$SOURCE_SUITE_ROOT/dataset/unified_eval_rows.csv}"
TRAIN_CSV="${SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_CSV:-$DATA_ROOT/unified_train_de_novo_rows.csv}"
EVAL_CSV="${SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_CSV:-$DATA_ROOT/unified_eval_de_novo_rows.csv}"
TRAIN_FEATURES_DIR="${SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR:-$SOURCE_SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR:-$SOURCE_SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
DIRECT_CHECKPOINT="${SUCC_UNIFIED_DIRECT_WARMSTART_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
BENCHMARK_TASKS="${SUCC_UNIFIED_DIRECT_WARMSTART_BENCHMARK_TASKS:-denovo_2p7p,denovo_ood}"
FILTER_BENCHMARK_CONTAINS="${SUCC_UNIFIED_DIRECT_WARMSTART_FILTER_BENCHMARK_CONTAINS:-}"

CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_FEATURE_VARIANT:-full}"
INPUT_MODALITY="${SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY:-with_image}"
CONDITION_LAYOUT="${SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT:-direct_compat}"
RUN_WARMSTART_BASELINE="${SUCC_UNIFIED_DIRECT_WARMSTART_RUN_BASELINE:-1}"

GRPO_DIR="${SUCC_UNIFIED_DIRECT_WARMSTART_GRPO_DIR:-$OUTPUT_ROOT/group_rl_grpo}"
WARMSTART_BENCH_DIR="${SUCC_UNIFIED_DIRECT_WARMSTART_BASELINE_BENCHMARK_DIR:-$OUTPUT_ROOT/benchmark_warmstart_beam}"
GRPO_BENCH_DIR="${SUCC_UNIFIED_DIRECT_WARMSTART_GRPO_BENCHMARK_DIR:-$OUTPUT_ROOT/benchmark_grpo_beam}"

BEAM_SIZE="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE:-40}"
BEAM_EXPAND_SIZE="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_EXPAND_SIZE:-128}"
BEAM_LENGTH_PENALTY="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_LENGTH_PENALTY:-0.8}"
TOP_K_CANDIDATES="${SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES:-40}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: missing required directory: $1" >&2
    exit 2
  fi
}

require_file "$SOURCE_TRAIN_CSV"
require_file "$SOURCE_EVAL_CSV"
require_file "$DIRECT_CHECKPOINT"
require_dir "$TRAIN_FEATURES_DIR"
require_dir "$EVAL_FEATURES_DIR"

mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
FILTER_ARGS=()
if [[ -n "$FILTER_BENCHMARK_CONTAINS" ]]; then
  FILTER_ARGS=(--benchmark-task-contains "$FILTER_BENCHMARK_CONTAINS")
fi

echo "Unified direct warm-start GRPO beam run"
echo "  source_suite_root=$SOURCE_SUITE_ROOT"
echo "  output_root=$OUTPUT_ROOT"
echo "  direct_checkpoint=$DIRECT_CHECKPOINT"
echo "  condition_feature_variant=$CONDITION_FEATURE_VARIANT"
echo "  condition_layout=$CONDITION_LAYOUT"
echo "  input_modality=$INPUT_MODALITY"
echo "  benchmark_tasks=$BENCHMARK_TASKS"
echo "  filter_benchmark_contains=${FILTER_BENCHMARK_CONTAINS:-none}"
echo "  beam_size=$BEAM_SIZE"
echo "  beam_expand_size=$BEAM_EXPAND_SIZE"

"$PYTHON_BIN" "$SCRIPT_DIR/filter_unified_smiles_rows.py" \
  --input-csv "$SOURCE_TRAIN_CSV" \
  --output-csv "$TRAIN_CSV" \
  --task-mode de_novo \
  "${FILTER_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/filter_unified_smiles_rows.py" \
  --input-csv "$SOURCE_EVAL_CSV" \
  --output-csv "$EVAL_CSV" \
  --task-mode de_novo \
  "${FILTER_ARGS[@]}"

run_beam_benchmark() {
  local checkpoint="$1"
  local output_dir="$2"
  local method="$3"
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
  SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$EVAL_FEATURES_DIR" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_dir" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$output_dir/sample_outputs" \
  SUCC_UNIFIED_BENCHMARK_TASKS="$BENCHMARK_TASKS" \
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="$CONDITION_FEATURE_VARIANT" \
  SUCC_UNIFIED_CONDITION_LAYOUT="$CONDITION_LAYOUT" \
  SUCC_UNIFIED_INPUT_MODALITY="$INPUT_MODALITY" \
  SUCC_UNIFIED_METHOD_NAME="$method" \
  SUCC_UNIFIED_DECODING_MODE="beam" \
  SUCC_UNIFIED_NUM_SAMPLES="1" \
  SUCC_UNIFIED_BEAM_SIZE="$BEAM_SIZE" \
  SUCC_UNIFIED_BEAM_EXPAND_SIZE="$BEAM_EXPAND_SIZE" \
  SUCC_UNIFIED_BEAM_LENGTH_PENALTY="$BEAM_LENGTH_PENALTY" \
  SUCC_UNIFIED_TOP_K_CANDIDATES="$TOP_K_CANDIDATES" \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

if [[ "$RUN_WARMSTART_BASELINE" == "1" ]]; then
  echo
  echo "=== Warm-start checkpoint beam sanity ==="
  run_beam_benchmark "$DIRECT_CHECKPOINT" "$WARMSTART_BENCH_DIR" "unified_direct_warmstart_beam"
fi

echo
echo "=== GRPO de novo training from direct checkpoint ==="
SUCC_UNIFIED_RL_TRAIN_CSV="$TRAIN_CSV" \
SUCC_UNIFIED_RL_EVAL_CSV="$EVAL_CSV" \
SUCC_UNIFIED_RL_OUTPUT_DIR="$GRPO_DIR" \
SUCC_UNIFIED_RL_RESUME_CHECKPOINT="$DIRECT_CHECKPOINT" \
SUCC_UNIFIED_RL_TRAIN_FEATURES_DIR="$TRAIN_FEATURES_DIR" \
SUCC_UNIFIED_RL_EVAL_FEATURES_DIR="$EVAL_FEATURES_DIR" \
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="$CONDITION_FEATURE_VARIANT" \
SUCC_UNIFIED_CONDITION_LAYOUT="$CONDITION_LAYOUT" \
SUCC_UNIFIED_INPUT_MODALITY="$INPUT_MODALITY" \
SUCC_UNIFIED_RL_OBJECTIVE="${SUCC_UNIFIED_RL_OBJECTIVE:-grpo}" \
SUCC_UNIFIED_RL_GRPO_CLIP_EPS="${SUCC_UNIFIED_RL_GRPO_CLIP_EPS:-0.2}" \
SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS="${SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS:-2}" \
SUCC_UNIFIED_RL_EPOCHS="${SUCC_UNIFIED_RL_EPOCHS:-1}" \
SUCC_UNIFIED_RL_BATCH_SIZE="${SUCC_UNIFIED_RL_BATCH_SIZE:-8}" \
SUCC_UNIFIED_RL_EVAL_BATCH_SIZE="${SUCC_UNIFIED_RL_EVAL_BATCH_SIZE:-32}" \
SUCC_UNIFIED_RL_LR="${SUCC_UNIFIED_RL_LR:-5e-7}" \
SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT="${SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT:-16}" \
SUCC_UNIFIED_RL_SFT_WEIGHT="${SUCC_UNIFIED_RL_SFT_WEIGHT:-1.0}" \
SUCC_UNIFIED_RL_REFERENCE_KL_WEIGHT="${SUCC_UNIFIED_RL_REFERENCE_KL_WEIGHT:-0.05}" \
SUCC_UNIFIED_RL_REWARD_MODE="${SUCC_UNIFIED_RL_REWARD_MODE:-property_strict}" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_group_rl.sh"

GRPO_CHECKPOINT="$GRPO_DIR/unified_smiles_generator_group_rl.pt"
require_file "$GRPO_CHECKPOINT"

echo
echo "=== GRPO checkpoint beam benchmark ==="
run_beam_benchmark "$GRPO_CHECKPOINT" "$GRPO_BENCH_DIR" "unified_direct_warmstart_grpo_beam"

echo
echo "Unified direct warm-start GRPO beam outputs:"
echo "  output_root=$OUTPUT_ROOT"
if [[ "$RUN_WARMSTART_BASELINE" == "1" ]]; then
  echo "  warmstart_report=$WARMSTART_BENCH_DIR/benchmark_suite_report.md"
fi
echo "  grpo_checkpoint=$GRPO_CHECKPOINT"
echo "  grpo_report=$GRPO_BENCH_DIR/benchmark_suite_report.md"
