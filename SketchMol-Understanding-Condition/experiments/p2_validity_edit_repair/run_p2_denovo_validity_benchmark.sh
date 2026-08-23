#!/usr/bin/env bash
# Compare legacy and syntax-safe decoding on one frozen P2 de novo subset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
BENCHMARK="${P2_BENCHMARK:-${1:-}}"
OUTPUT_ROOT="${P2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p2_validity_edit_repair_seed7}"
NUM_SAMPLES="${P2_NUM_SAMPLES:-20}"
SEED="${P2_EVAL_SEED:-20260823}"

case "$BENCHMARK" in
  two_p_to_seven_p)
    BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
    CHECKPOINT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt"
    EVAL_CSV="$OUTPUT_ROOT/data/denovo_2p7p_eval.csv"
    LEGACY_T=0.85; LEGACY_K=40; LEGACY_P=0.95; LEGACY_R=1.15
    SAFE_T=0.85; SAFE_K=40; SAFE_P=0.95; SAFE_R=1.05
    ;;
  ood)
    BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_mixed_condition"
    CHECKPOINT="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt"
    EVAL_CSV="$OUTPUT_ROOT/data/denovo_ood_eval.csv"
    LEGACY_T=0.70; LEGACY_K=24; LEGACY_P=0.90; LEGACY_R=1.20
    SAFE_T=0.70; SAFE_K=24; SAFE_P=0.90; SAFE_R=1.10
    ;;
  *)
    echo "Usage: P2_BENCHMARK={two_p_to_seven_p|ood} $0" >&2
    exit 2
    ;;
esac

for required in "$CHECKPOINT" "$EVAL_CSV" "$BASE_DIR/train_condition_features_hf_vlm" "$BASE_DIR/eval_condition_features_hf_vlm"; do
  [[ -e "$required" ]] || { echo "ERROR: missing P2 input: $required" >&2; exit 2; }
done

run_arm() {
  local arm="$1"
  local temperature="$2"
  local top_k="$3"
  local top_p="$4"
  local repetition_penalty="$5"
  local no_repeat_ngram="$6"
  local grammar="$7"
  local arm_dir="$OUTPUT_ROOT/denovo/$BENCHMARK/$arm"
  local grammar_args=()
  [[ "$grammar" == "1" ]] && grammar_args+=(--smiles-grammar-constraint)
  if [[ -f "$arm_dir/COMPLETE" && -s "$arm_dir/raw_candidates_n${NUM_SAMPLES}.csv" ]]; then
    echo "P2 arm already complete: $BENCHMARK/$arm"
    return
  fi
  mkdir -p "$arm_dir/model_eval"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
    --eval-only \
    --eval-csv "$EVAL_CSV" \
    --resume-checkpoint "$CHECKPOINT" \
    --condition-features-dir "$BASE_DIR/train_condition_features_hf_vlm" \
    --eval-condition-features-dir "$BASE_DIR/eval_condition_features_hf_vlm" \
    --condition-mixing-mode append_property_program \
    --output-dir "$arm_dir/model_eval" \
    --prediction-csv "$arm_dir/diagnostic_property_reranked_selected.csv" \
    --candidate-output-csv "$arm_dir/raw_candidates_n${NUM_SAMPLES}.csv" \
    --eval-batch-size "${P2_EVAL_BATCH_SIZE:-32}" \
    --max-new-tokens "${P2_MAX_NEW_TOKENS:-96}" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --num-samples "$NUM_SAMPLES" \
    --parallel-samples "${P2_PARALLEL_SAMPLES:-8}" \
    --max-parallel-sequences "${P2_MAX_PARALLEL_SEQUENCES:-1024}" \
    --repetition-penalty "$repetition_penalty" \
    --no-repeat-ngram-size "$no_repeat_ngram" \
    --min-new-tokens 6 \
    --seed "$SEED" \
    --device auto \
    "${grammar_args[@]}"
  touch "$arm_dir/COMPLETE"
}

run_arm legacy "$LEGACY_T" "$LEGACY_K" "$LEGACY_P" "$LEGACY_R" 6 0
run_arm syntax_safe "$SAFE_T" "$SAFE_K" "$SAFE_P" "$SAFE_R" 0 1

echo "P2 de novo benchmark complete: $BENCHMARK"
