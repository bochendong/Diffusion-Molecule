#!/usr/bin/env bash
# Submit fair-budget direct-SMILES de novo evaluations for SketchMol comparison.
#
# Defaults submit Ours@1 and Ours@40 for both 2p-7p and OOD using the current
# group-RL checkpoints. This is inference-only and does not retrain models.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

SUITE_ROOT="${SUCC_DIRECT_FAIR_BUDGET_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_v2_fair_budget_suite_v1}"
BUDGETS="${SUCC_DIRECT_FAIR_BUDGETS:-1,40}"
RUN_2P7P="${SUCC_DIRECT_FAIR_BUDGET_RUN_2P7P:-1}"
RUN_OOD="${SUCC_DIRECT_FAIR_BUDGET_RUN_OOD:-1}"
TIME_N1="${SUCC_DIRECT_FAIR_BUDGET_TIME_N1:-02:00:00}"
TIME_N40="${SUCC_DIRECT_FAIR_BUDGET_TIME_N40:-06:00:00}"
TIME_DEFAULT="${SUCC_DIRECT_FAIR_BUDGET_TIME_DEFAULT:-08:00:00}"
GROUP_RL_2P7P_OUTPUT="${SUCC_DIRECT_FAIR_BUDGET_2P7P_GROUP_RL_OUTPUT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1}"
GROUP_RL_OOD_OUTPUT="${SUCC_DIRECT_FAIR_BUDGET_OOD_GROUP_RL_OUTPUT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_group_rl_v1}"
GROUP_RL_2P7P_CKPT="${SUCC_DIRECT_FAIR_BUDGET_2P7P_CKPT:-$GROUP_RL_2P7P_OUTPUT/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
GROUP_RL_OOD_CKPT="${SUCC_DIRECT_FAIR_BUDGET_OOD_CKPT:-$GROUP_RL_OOD_OUTPUT/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"

if [[ "$RUN_2P7P" == "1" && ! -f "$GROUP_RL_2P7P_CKPT" ]]; then
  echo "ERROR: missing 2p-7p group-RL checkpoint: $GROUP_RL_2P7P_CKPT" >&2
  exit 2
fi
if [[ "$RUN_OOD" == "1" && ! -f "$GROUP_RL_OOD_CKPT" ]]; then
  echo "ERROR: missing OOD group-RL checkpoint: $GROUP_RL_OOD_CKPT" >&2
  exit 2
fi

mkdir -p "$SUITE_ROOT"

choose_time() {
  local budget="$1"
  if [[ "$budget" == "1" ]]; then
    echo "$TIME_N1"
  elif [[ "$budget" == "40" ]]; then
    echo "$TIME_N40"
  else
    echo "$TIME_DEFAULT"
  fi
}

submit_2p7p() {
  local budget="$1"
  local run_dir="$SUITE_ROOT/2p7p_n${budget}"
  local time
  time="$(choose_time "$budget")"
  echo
  echo "=== submitting fair 2p-7p Ours@${budget} ==="
  local output
  if ! output="$(
    SUCC_DIRECT_GROUP_RL_RUN_TRAIN=0 \
    SUCC_DIRECT_GROUP_RL_RL_CHECKPOINT="$GROUP_RL_2P7P_CKPT" \
    SUCC_DIRECT_GROUP_RL_OUTPUT_DIR="$run_dir" \
    SUCC_DIRECT_GROUP_RL_BENCHMARK_OUTPUT_DIR="$run_dir/benchmark_direct_smiles_group_rl_n${budget}" \
    SUCC_DIRECT_GROUP_RL_BENCHMARK_MODEL_DIR="$run_dir/direct_smiles_model_group_rl_eval_n${budget}" \
    SUCC_DIRECT_GROUP_RL_BENCHMARK_PREDICTION_CSV="$run_dir/benchmark_direct_smiles_group_rl_n${budget}/direct_smiles_predictions.csv" \
    SUCC_DIRECT_GROUP_RL_BENCHMARK_NUM_SAMPLES="$budget" \
    SUCC_DIRECT_GROUP_RL_SLURM_JOB_NAME="succ-denovo-fair-2p7p-n${budget}" \
    SUCC_DIRECT_GROUP_RL_SLURM_TIME="$time" \
    bash "$PROJECT_DIR/scripts/submit_direct_smiles_group_rl_2p7p_v2.sh" 2>&1
  )"; then
    echo "$output"
    echo "ERROR: failed to submit 2p-7p Ours@${budget}" >&2
    exit 1
  fi
  echo "$output"
}

submit_ood() {
  local budget="$1"
  local run_dir="$SUITE_ROOT/ood_n${budget}"
  local time
  time="$(choose_time "$budget")"
  echo
  echo "=== submitting fair OOD Ours@${budget} ==="
  local output
  if ! output="$(
    SUCC_DIRECT_OOD_GROUP_RL_RUN_TRAIN=0 \
    SUCC_DIRECT_OOD_GROUP_RL_RL_CHECKPOINT="$GROUP_RL_OOD_CKPT" \
    SUCC_DIRECT_OOD_GROUP_RL_OUTPUT_DIR="$run_dir" \
    SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_OUTPUT_DIR="$run_dir/benchmark_direct_smiles_group_rl_n${budget}" \
    SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_MODEL_DIR="$run_dir/direct_smiles_model_group_rl_eval_n${budget}" \
    SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_PREDICTION_CSV="$run_dir/benchmark_direct_smiles_group_rl_n${budget}/direct_smiles_predictions.csv" \
    SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_NUM_SAMPLES="$budget" \
    SUCC_DIRECT_OOD_GROUP_RL_SLURM_JOB_NAME="succ-denovo-fair-ood-n${budget}" \
    SUCC_DIRECT_OOD_GROUP_RL_SLURM_TIME="$time" \
    bash "$PROJECT_DIR/scripts/submit_direct_smiles_group_rl_ood_v2.sh" 2>&1
  )"; then
    echo "$output"
    echo "ERROR: failed to submit OOD Ours@${budget}" >&2
    exit 1
  fi
  echo "$output"
}

echo "Direct-SMILES de novo fair-budget suite"
echo "  suite_root=$SUITE_ROOT"
echo "  budgets=$BUDGETS"
echo "  run_2p7p=$RUN_2P7P"
echo "  run_ood=$RUN_OOD"
echo "  2p7p_ckpt=$GROUP_RL_2P7P_CKPT"
echo "  ood_ckpt=$GROUP_RL_OOD_CKPT"

IFS=',' read -r -a budget_array <<< "$BUDGETS"
for budget in "${budget_array[@]}"; do
  budget="${budget#"${budget%%[![:space:]]*}"}"
  budget="${budget%"${budget##*[![:space:]]}"}"
  [[ -z "$budget" ]] && continue
  if [[ "$RUN_2P7P" == "1" ]]; then
    submit_2p7p "$budget"
  fi
  if [[ "$RUN_OOD" == "1" ]]; then
    submit_ood "$budget"
  fi
done

echo
echo "Fair-budget suite submitted."
echo "  suite_root=$SUITE_ROOT"
echo "  collect_when_done=python SketchMol-Understanding-Condition/scripts/collect_direct_smiles_denovo_fair_budget_results.py --suite-root $SUITE_ROOT"
