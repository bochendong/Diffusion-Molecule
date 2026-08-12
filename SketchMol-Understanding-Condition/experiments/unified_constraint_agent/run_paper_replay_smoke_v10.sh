#!/usr/bin/env bash
# Small, frozen De novo 2p-7p and MolEdit Table1 replay for anti-forgetting signal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

STAGE="${1:-all}"
if [[ "$STAGE" != "prepare" && "$STAGE" != "rank" && "$STAGE" != "score" && "$STAGE" != "all" ]]; then
  echo "ERROR: stage must be prepare, rank, score, or all" >&2
  exit 2
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_PAPER_REPLAY_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_paper_replay_smoke_v10/seed_1713}"
SMOKE_DIR="$RUN_ROOT/smoke"
DENOVO_DIR="$RUN_ROOT/denovo_n20"
STABLE_ROOT="$RUN_ROOT/table1_stable"
RESIDUAL_ROOT="$RUN_ROOT/table1_residual"
STABLE_ADAPTER="${SUCC_UCA_STABLE_ADAPTER:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter}"
RESIDUAL_ADAPTER="${SUCC_UCA_RESIDUAL_ADAPTER:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_residual_planner_v9/seed_1712/model/adapter}"
DENOVO_EVAL="${SUCC_UCA_DENOVO_EVAL:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv}"
DENOVO_CANDIDATES="${SUCC_UCA_DENOVO_CANDIDATES:-$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_zero_safe_n40_v1/benchmark_direct_smiles_group_rl/direct_smiles_candidates.csv}"
TABLE1_INPUT="${SUCC_UCA_TABLE1_INPUT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="$GSK3B_ORACLE"
mkdir -p "$RUN_ROOT"

for path in "$DENOVO_EVAL" "$DENOVO_CANDIDATES" "$TABLE1_INPUT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing replay input: $path" >&2; exit 2; }
done

prepare() {
  "$PYTHON_BIN" "$SCRIPT_DIR/build_paper_replay_smoke_rows.py" \
    --denovo-eval-csv "$DENOVO_EVAL" \
    --denovo-candidate-csv "$DENOVO_CANDIDATES" \
    --table1-input-csv "$TABLE1_INPUT" \
    --output-dir "$SMOKE_DIR" \
    --denovo-per-bucket 10 \
    --table1-per-task 10 \
    --candidate-budget 20 \
    --seed 1713

  "$PYTHON_BIN" "$REPO_DIR/SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py" \
    --eval-csv "$SMOKE_DIR/denovo_eval.csv" \
    --candidate-csv "$SMOKE_DIR/denovo_candidates_n20.csv" \
    --output-dir "$DENOVO_DIR" \
    --budgets 20 \
    --report-title "GraphEdit unified paper replay smoke: De novo 2p-7p exact n=20" \
    --candidate-description "frozen first-20 Direct-SMILES Group-RL candidates"
}

rank_one() {
  local label="$1"
  local adapter="$2"
  local output_root="$3"
  [[ -f "$adapter/adapter_model.safetensors" ]] || { echo "ERROR: missing adapter: $adapter" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_official_actions.py" \
    --input-csv "$SMOKE_DIR/table1_rows.csv" \
    --output-dir "$output_root/ranking" \
    --suite table1 \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-dir "$adapter" \
    --variant "$label" \
    --candidate-budget 20 \
    --verifier-k 5 \
    --enumeration-attempt-budget 64 \
    --max-enumeration-attempt-budget 512 \
    --site-limit 32 \
    --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}" \
    --max-length 1024
}

rank() {
  rank_one stable_seed_1703_smoke "$STABLE_ADAPTER" "$STABLE_ROOT"
  rank_one residual_v9_seed_1712_smoke "$RESIDUAL_ADAPTER" "$RESIDUAL_ROOT"
}

score_one() {
  local label="$1"
  local output_root="$2"
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
  SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$output_root/ranking/candidates.csv" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_root/official" \
  SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
  SUCC_UNIFIED_METHOD_NAME="$label" \
  SUCC_UNIFIED_CANDIDATE_BUDGETS=1,5,20 \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$SMOKE_DIR/table1_rows.csv" \
  SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
  SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY=fail \
  SUCC_GSK3B_ORACLE_PATH="$GSK3B_ORACLE" \
  bash "$SCRIPT_DIR/../unified_smiles_generator/run_unified_smiles_generator_benchmark_suite.sh"
}

score() {
  score_one stable_seed_1703_smoke "$STABLE_ROOT"
  score_one residual_v9_seed_1712_smoke "$RESIDUAL_ROOT"
  "$PYTHON_BIN" "$SCRIPT_DIR/compare_paper_replay_smoke.py" \
    --smoke-manifest "$SMOKE_DIR/manifest.json" \
    --denovo-summary "$DENOVO_DIR/budget_sweep_summary.csv" \
    --stable-ranking-summary "$STABLE_ROOT/ranking/summary.json" \
    --residual-ranking-summary "$RESIDUAL_ROOT/ranking/summary.json" \
    --stable-table-root "$STABLE_ROOT/official" \
    --residual-table-root "$RESIDUAL_ROOT/official" \
    --stable-candidates "$STABLE_ROOT/ranking/candidates.csv" \
    --residual-candidates "$RESIDUAL_ROOT/ranking/candidates.csv" \
    --output-dir "$RUN_ROOT/gate" \
    --max-table1-drop 0.02
}

case "$STAGE" in
  prepare) prepare ;;
  rank) rank ;;
  score) score ;;
  all) prepare; rank; score ;;
esac

echo "Paper replay smoke stage complete: stage=$STAGE root=$RUN_ROOT"
