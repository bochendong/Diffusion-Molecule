#!/usr/bin/env bash
# Generate one P1 6p/7p n=20 kill-test arm with frozen checkpoints and decoding.

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
ARM="${P1_FAST_ARM:-${1:-}}"
BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
OUTPUT_ROOT="${SUCC_P1_FAST_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p1_fast_hard_6p7p_seed7}"
EVAL_CSV="$OUTPUT_ROOT/eval_6p7p_128_each.csv"

case "$ARM" in
  sft) CHECKPOINT="$BASE_DIR/direct_smiles_model/direct_smiles_generator.pt" ;;
  group_rl) CHECKPOINT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt" ;;
  *) echo "Usage: P1_FAST_ARM={sft|group_rl} $0" >&2; exit 2 ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$ARM"
CANDIDATE_CSV="$OUTPUT_DIR/raw_candidates_n20.csv"
mkdir -p "$OUTPUT_DIR/model_eval"
if [[ -f "$OUTPUT_DIR/COMPLETE" && -s "$CANDIDATE_CSV" ]]; then
  echo "P1 fast arm already complete: $ARM"
  exit 0
fi
if [[ "$ARM" == "group_rl" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/validate_p1_recovered_checkpoint.py" --checkpoint "$CHECKPOINT" --benchmark two_p_to_seven_p --output-json "$OUTPUT_DIR/recovered_checkpoint_validation.json"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
  --eval-only --eval-csv "$EVAL_CSV" --resume-checkpoint "$CHECKPOINT" \
  --condition-features-dir "$BASE_DIR/train_condition_features_hf_vlm" \
  --eval-condition-features-dir "$BASE_DIR/eval_condition_features_hf_vlm" \
  --condition-mixing-mode append_property_program --output-dir "$OUTPUT_DIR/model_eval" \
  --prediction-csv "$OUTPUT_DIR/diagnostic_property_reranked_selected.csv" \
  --candidate-output-csv "$CANDIDATE_CSV" --eval-batch-size 32 --max-new-tokens 96 \
  --temperature 0.85 --top-k 40 --top-p 0.95 --num-samples 20 --parallel-samples 8 \
  --max-parallel-sequences 1024 --repetition-penalty 1.15 --no-repeat-ngram-size 6 \
  --min-new-tokens 6 --seed 7 --device auto

touch "$OUTPUT_DIR/COMPLETE"
echo "P1 fast arm complete: $ARM"
