#!/usr/bin/env bash
# Train and gate a stable-anchor, top-k-preserving common-LLM residual adapter.

set -euo pipefail

STAGE="${1:?usage: run_anchor_residual_v11.sh prepare|gpu|gpu_rank|oracle_gate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
SFT_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1"
VERIFIER_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_verifier_preference_v2/data/seed_1705"
V9_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_residual_planner_v9/seed_1712"
TABLE1_SMOKE="$PROJECT_DIR/outputs/unified_constraint_agent_paper_replay_smoke_v10/seed_1713/smoke/table1_rows.csv"
RUN_ROOT="${SUCC_UCA_ANCHOR_RESIDUAL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_anchor_residual_v11/seed_1714}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
STABLE_ADAPTER="$SFT_ROOT/model/seed_1703/adapter"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"
RAW_PREF="$RUN_ROOT/data/combined_preferences"
REF_PREF="$RUN_ROOT/data/reference_annotated_preferences"
MODEL_DIR="$RUN_ROOT/model"
PREF_EVAL="$RUN_ROOT/preference_eval"
TABLE1_DIR="$RUN_ROOT/table1_smoke"
MUMO_DIR="$RUN_ROOT/mumo_dev"
ORACLE_DIR="$RUN_ROOT/oracle"
FORGETTING_DIR="$RUN_ROOT/anti_forgetting"
GATE_DIR="$RUN_ROOT/gate"

BASELINE_CANDIDATES="$V9_ROOT/candidate_pool/deterministic_n20.csv"
ENUMERATED_CANDIDATES="$V9_ROOT/candidate_pool/internal_top48.csv"
V9_PREFERENCE="$V9_ROOT/data/residual_preferences"
BASELINE_GATE="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711/gate/summary.json"
BASELINE_ORACLE="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711/oracle/generated_properties.csv"

export PYTHONPATH="$DEP_OVERLAY:$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="$GSK3B_ORACLE"
mkdir -p "$RUN_ROOT"

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" "$SCRIPT_DIR/build_anchor_residual_preferences.py" \
      --verifier-preference-dir "$VERIFIER_ROOT" \
      --mumo-residual-preference-dir "$V9_PREFERENCE" \
      --output-dir "$RAW_PREF" \
      --max-table1-pairs 256 --max-mumo-graph-pairs 72 --max-mumo-residual-pairs 256 \
      --seed 1714
    ;;
  gpu)
    "$PYTHON_BIN" "$SCRIPT_DIR/annotate_preference_reference_scores.py" \
      --input-train-jsonl "$RAW_PREF/train.jsonl" \
      --input-validation-jsonl "$RAW_PREF/validation.jsonl" \
      --output-dir "$REF_PREF" --base-model "$BASE_MODEL" \
      --adapter-dir "$STABLE_ADAPTER" --batch-size 4 --max-length 512
    "$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_preference.py" \
      --train-jsonl "$REF_PREF/train.jsonl" \
      --validation-jsonl "$REF_PREF/validation.jsonl" \
      --input-adapter-dir "$STABLE_ADAPTER" --output-dir "$MODEL_DIR" \
      --base-model "$BASE_MODEL" --max-length 512 --epochs 1 --batch-size 1 \
      --gradient-accumulation 8 --learning-rate 2e-6 --beta 2.0 --sft-weight 0.05 \
      --reference-margin-field stable_reference_margin \
      --replay-jsonl "$SFT_ROOT/data/common_llm_sft/train.jsonl" \
      --replay-sft-weight 0.15 --replay-batch-size 1 --replay-max-per-origin 256 \
      --seed 1714
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_preferences.py" \
      --input-jsonl "$REF_PREF/validation.jsonl" --output-json "$PREF_EVAL/candidate.json" \
      --base-model "$BASE_MODEL" --adapter-dir "$MODEL_DIR/adapter" \
      --variant anchor_residual_v11 --batch-size 4 --max-length 512
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_pilot.py" \
      --input-jsonl "$SFT_ROOT/data/common_llm_sft/validation.jsonl" \
      --output-dir "$FORGETTING_DIR/candidate" --base-model "$BASE_MODEL" \
      --adapter-dir "$MODEL_DIR/adapter" --variant anchor_residual_v11 \
      --batch-size 8 --max-new-tokens 128
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_official_actions.py" \
      --input-csv "$TABLE1_SMOKE" --output-dir "$TABLE1_DIR/ranking" --suite table1 \
      --base-model "$BASE_MODEL" --adapter-dir "$MODEL_DIR/adapter" \
      --reference-adapter-dir "$STABLE_ADAPTER" --anchor-top-k 5 \
      --max-residual-rank-shift 12 --variant anchor_residual_v11 \
      --candidate-budget 20 --verifier-k 5 --enumeration-attempt-budget 64 \
      --max-enumeration-attempt-budget 512 --site-limit 32 --score-batch-size 4 --max-length 1024
    "$PYTHON_BIN" "$SCRIPT_DIR/rank_mumo_residual_candidates.py" \
      --baseline-csv "$BASELINE_CANDIDATES" --enumerated-csv "$ENUMERATED_CANDIDATES" \
      --output-csv "$MUMO_DIR/candidates.csv" --manifest-json "$MUMO_DIR/manifest.json" \
      --base-model "$BASE_MODEL" --adapter-dir "$MODEL_DIR/adapter" \
      --reference-adapter-dir "$STABLE_ADAPTER" --preference-manifest "$RAW_PREF/manifest.json" \
      --baseline-prefix 15 --residual-slots 5 --max-llm-rank-shift 12 \
      --score-batch-size 16 --max-length 512 --method-name common_llm_anchor_residual_v11 \
      --progress-jsonl "$MUMO_DIR/ranking_progress.jsonl"
    ;;
  gpu_rank)
    [[ -s "$MODEL_DIR/adapter/adapter_model.safetensors" ]] || {
      echo "ERROR: completed v11 adapter is missing" >&2; exit 2;
    }
    [[ -s "$PREF_EVAL/candidate.json" && -s "$FORGETTING_DIR/candidate/summary.json" ]] || {
      echo "ERROR: completed v11 evaluation artifacts are missing" >&2; exit 2;
    }
    [[ -s "$TABLE1_DIR/ranking/summary.json" && -s "$TABLE1_DIR/ranking/candidates.csv" ]] || {
      echo "ERROR: completed v11 Table1 artifacts are missing" >&2; exit 2;
    }
    "$PYTHON_BIN" "$SCRIPT_DIR/rank_mumo_residual_candidates.py" \
      --baseline-csv "$BASELINE_CANDIDATES" --enumerated-csv "$ENUMERATED_CANDIDATES" \
      --output-csv "$MUMO_DIR/candidates.csv" --manifest-json "$MUMO_DIR/manifest.json" \
      --base-model "$BASE_MODEL" --adapter-dir "$MODEL_DIR/adapter" \
      --reference-adapter-dir "$STABLE_ADAPTER" --preference-manifest "$RAW_PREF/manifest.json" \
      --baseline-prefix 15 --residual-slots 5 --max-llm-rank-shift 12 \
      --score-batch-size 16 --max-length 512 --method-name common_llm_anchor_residual_v11 \
      --progress-jsonl "$MUMO_DIR/ranking_progress.jsonl"
    ;;
  oracle_gate)
    SUCC_PYTHON_BIN="$PYTHON_BIN" \
    SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
    SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$TABLE1_DIR/ranking/candidates.csv" \
    SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$TABLE1_DIR/official" \
    SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
    SUCC_UNIFIED_METHOD_NAME=common_llm_anchor_residual_v11 \
    SUCC_UNIFIED_CANDIDATE_BUDGETS=1,5,20 \
    SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
    SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$TABLE1_SMOKE" \
    SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
    SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY=fail \
    SUCC_GSK3B_ORACLE_PATH="$GSK3B_ORACLE" \
    bash "$SCRIPT_DIR/../unified_smiles_generator/run_unified_smiles_generator_benchmark_suite.sh"
    SUCC_PYTHON_BIN="$PYTHON_BIN" SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$MUMO_DIR/candidates.csv" \
    SUCC_ORACLE_OUTPUT_CSV="$ORACLE_DIR/generated_properties.csv" \
    SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
    SUCC_ORACLE_MERGE_PROPERTIES_CSV="$BASELINE_ORACLE" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES=bbbp,hia,mutagenicity \
    bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
      --prediction-csv "$MUMO_DIR/candidates.csv" --output-dir "$MUMO_DIR/evaluation" \
      --generated-properties-csv "$ORACLE_DIR/generated_properties.csv" \
      --source-properties-csv "$ORACLE_DIR/generated_properties.csv" \
      --group-column condition_id --min-source-tanimoto 0.4 \
      --report-title "Stable-anchor residual v11 MuMO dev n=20"
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_mumo_residual_gate.py" \
      --candidate-manifest "$MUMO_DIR/manifest.json" \
      --summary-csv "$MUMO_DIR/evaluation/external_multiproperty_summary.csv" \
      --oracle-summary "$ORACLE_DIR/generated_properties.summary.json" \
      --baseline-gate "$BASELINE_GATE" \
      --baseline-format-summary "$V9_ROOT/anti_forgetting/baseline/summary.json" \
      --candidate-format-summary "$FORGETTING_DIR/candidate/summary.json" \
      --baseline-preference-summary "$PREF_EVAL/candidate.json" \
      --candidate-preference-summary "$PREF_EVAL/candidate.json" \
      --preference-manifest "$RAW_PREF/manifest.json" \
      --training-summary "$MODEL_DIR/training_summary.json" --output-dir "$MUMO_DIR/gate" \
      --min-preference-accuracy 0.0
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_anchor_residual_gate.py" \
      --table1-summary "$TABLE1_DIR/ranking/summary.json" \
      --table1-official-root "$TABLE1_DIR/official" \
      --mumo-gate "$MUMO_DIR/gate/summary.json" \
      --preference-manifest "$RAW_PREF/manifest.json" \
      --reference-annotation-summary "$REF_PREF/summary.json" \
      --training-summary "$MODEL_DIR/training_summary.json" \
      --output-dir "$GATE_DIR"
    ;;
  *) echo "ERROR: unknown stage $STAGE" >&2; exit 2 ;;
esac
