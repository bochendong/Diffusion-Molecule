#!/usr/bin/env bash
# Train, evaluate, and gate a bounded 1.5B residual planner at exact MuMO n=20.

set -euo pipefail

STAGE="${1:?usage: run_mumo_residual_planner_v9.sh prepare|enumerate|merge|gpu|oracle_gate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
EVIDENCE_ROOT="${SUCC_UCA_MUMO_PARALLEL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711}"
CLOSED_ROOT="${SUCC_UCA_MUMO_CLOSED_LOOP_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711}"
SFT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
RUN_ROOT="${SUCC_UCA_MUMO_RESIDUAL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_residual_planner_v9/seed_1712}"
SEED="${SUCC_UCA_SEED:-1712}"
SHARD_COUNT="${SUCC_UCA_MUMO_DEV_SHARD_COUNT:-16}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

DEV_SOURCES="$CLOSED_ROOT/data/dev_sources.jsonl"
DEV_SOURCE_MANIFEST="$CLOSED_ROOT/data/dev_sources.manifest.json"
BASELINE_GATE="$CLOSED_ROOT/gate/summary.json"
BASELINE_ORACLE="$CLOSED_ROOT/oracle/generated_properties.csv"
SFT_DATA="$SFT_ROOT/data/common_llm_sft"
INPUT_ADAPTER="${SUCC_UCA_INPUT_ADAPTER:-$SFT_ROOT/model/seed_1703/adapter}"
PREFERENCE_DIR="$RUN_ROOT/data/residual_preferences"
POOL_DIR="$RUN_ROOT/candidate_pool"
BASELINE_CANDIDATES="$POOL_DIR/deterministic_n20.csv"
ENUMERATED_CANDIDATES="$POOL_DIR/internal_top48.csv"
POOL_MANIFEST="$POOL_DIR/manifest.json"
MODEL_DIR="$RUN_ROOT/model"
FORGETTING_DIR="$RUN_ROOT/anti_forgetting"
PREFERENCE_EVAL_DIR="$RUN_ROOT/preference_eval"
PLANNER_DIR="$RUN_ROOT/planner"
PLANNER_CANDIDATES="$PLANNER_DIR/residual_n20.csv"
PLANNER_MANIFEST="$PLANNER_DIR/manifest.json"
ORACLE_DIR="$RUN_ROOT/oracle"
ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
EVAL_DIR="$RUN_ROOT/evaluation"
GATE_DIR="$RUN_ROOT/gate"

export PYTHONPATH="$DEP_OVERLAY:$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PREFERENCE_DIR" "$POOL_DIR/shards" "$MODEL_DIR" "$FORGETTING_DIR" \
  "$PREFERENCE_EVAL_DIR" "$PLANNER_DIR" "$ORACLE_DIR" "$EVAL_DIR" "$GATE_DIR"

for path in "$PYTHON_BIN" "$DEV_SOURCES" "$DEV_SOURCE_MANIFEST" "$BASELINE_GATE" \
  "$SFT_DATA/train.jsonl" "$SFT_DATA/validation.jsonl" "$INPUT_ADAPTER/adapter_model.safetensors"; do
  [[ -e "$path" ]] || { echo "ERROR: missing v9 input: $path" >&2; exit 2; }
done
"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True; assert p["candidate_budget"] == 20; assert p["evaluation_target_access"] is False' "$BASELINE_GATE"

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" "$SCRIPT_DIR/build_mumo_residual_preferences.py" \
      --data-dir "$EVIDENCE_ROOT/data" \
      --evidence-root "$EVIDENCE_ROOT" \
      --output-dir "$PREFERENCE_DIR" \
      --max-conditions-per-task "${SUCC_UCA_RESIDUAL_TRAIN_PER_TASK:-100}" \
      --validation-fraction 0.10 \
      --seed "$SEED"
    ;;
  enumerate)
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?enumerate requires a Slurm array task}"
    SHARD_TAG="$(printf '%03d' "$SHARD_INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_mumo_closed_loop_dev.py" \
      --run-root "$EVIDENCE_ROOT" \
      --dev-sources-jsonl "$DEV_SOURCES" \
      --output-csv "$POOL_DIR/shards/candidates_${SHARD_TAG}.csv" \
      --enumerated-output-csv "$POOL_DIR/shards/enumerated_${SHARD_TAG}.csv" \
      --manifest-json "$POOL_DIR/shards/manifest_${SHARD_TAG}.json" \
      --candidate-budget 20 \
      --planner-candidate-limit 48 \
      --shard-index "$SHARD_INDEX" \
      --shard-count "$SHARD_COUNT"
    ;;
  merge)
    EXPECTED_CONDITIONS="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$DEV_SOURCE_MANIFEST")"
    "$PYTHON_BIN" "$SCRIPT_DIR/merge_mumo_closed_loop_dev.py" \
      --shard-dir "$POOL_DIR/shards" \
      --output-csv "$BASELINE_CANDIDATES" \
      --enumerated-output-csv "$ENUMERATED_CANDIDATES" \
      --manifest-json "$POOL_MANIFEST" \
      --shard-count "$SHARD_COUNT" \
      --expected-conditions "$EXPECTED_CONDITIONS"
    ;;
  gpu)
    for path in "$PREFERENCE_DIR/train.jsonl" "$PREFERENCE_DIR/validation.jsonl" \
      "$PREFERENCE_DIR/manifest.json" "$BASELINE_CANDIDATES" "$ENUMERATED_CANDIDATES"; do
      [[ -s "$path" ]] || { echo "ERROR: missing v9 GPU input: $path" >&2; exit 2; }
    done
    "$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_preference.py" \
      --train-jsonl "$PREFERENCE_DIR/train.jsonl" \
      --validation-jsonl "$PREFERENCE_DIR/validation.jsonl" \
      --input-adapter-dir "$INPUT_ADAPTER" \
      --output-dir "$MODEL_DIR" \
      --base-model "$BASE_MODEL" \
      --max-length 512 \
      --epochs 1 \
      --batch-size 1 \
      --gradient-accumulation 8 \
      --learning-rate "${SUCC_UCA_RESIDUAL_LR:-3e-6}" \
      --beta 2.0 \
      --sft-weight 0.10 \
      --replay-jsonl "$SFT_DATA/train.jsonl" \
      --replay-sft-weight 0.10 \
      --replay-batch-size 1 \
      --replay-max-per-origin 256 \
      --seed "$SEED"
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_preferences.py" \
      --input-jsonl "$PREFERENCE_DIR/validation.jsonl" \
      --output-json "$PREFERENCE_EVAL_DIR/baseline.json" \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$INPUT_ADAPTER" \
      --variant stable_sft_seed_1703 \
      --batch-size 16
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_preferences.py" \
      --input-jsonl "$PREFERENCE_DIR/validation.jsonl" \
      --output-json "$PREFERENCE_EVAL_DIR/residual.json" \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$MODEL_DIR/adapter" \
      --variant mumo_residual_v9 \
      --batch-size 16
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_pilot.py" \
      --input-jsonl "$SFT_DATA/validation.jsonl" \
      --output-dir "$FORGETTING_DIR/baseline" \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$INPUT_ADAPTER" \
      --variant stable_sft_seed_1703 \
      --batch-size 8 \
      --max-new-tokens 128
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_pilot.py" \
      --input-jsonl "$SFT_DATA/validation.jsonl" \
      --output-dir "$FORGETTING_DIR/residual" \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$MODEL_DIR/adapter" \
      --variant mumo_residual_v9 \
      --batch-size 8 \
      --max-new-tokens 128
    "$PYTHON_BIN" "$SCRIPT_DIR/rank_mumo_residual_candidates.py" \
      --baseline-csv "$BASELINE_CANDIDATES" \
      --enumerated-csv "$ENUMERATED_CANDIDATES" \
      --output-csv "$PLANNER_CANDIDATES" \
      --manifest-json "$PLANNER_MANIFEST" \
      --base-model "$BASE_MODEL" \
      --adapter-dir "$MODEL_DIR/adapter" \
      --preference-manifest "$PREFERENCE_DIR/manifest.json" \
      --baseline-prefix 15 \
      --residual-slots 5 \
      --max-llm-rank-shift 12 \
      --score-batch-size 16 \
      --max-length 512
    ;;
  oracle_gate)
    [[ -s "$PLANNER_CANDIDATES" && -s "$PLANNER_MANIFEST" ]] || { echo "ERROR: residual n=20 pool missing" >&2; exit 2; }
    SUCC_PYTHON_BIN="$PYTHON_BIN" \
    SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$PLANNER_CANDIDATES" \
    SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
    SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
    SUCC_ORACLE_MERGE_PROPERTIES_CSV="$BASELINE_ORACLE" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
    bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
      --prediction-csv "$PLANNER_CANDIDATES" \
      --output-dir "$EVAL_DIR" \
      --generated-properties-csv "$ORACLE_CSV" \
      --source-properties-csv "$ORACLE_CSV" \
      --group-column condition_id \
      --min-source-tanimoto 0.4 \
      --report-title "Common 1.5B bounded residual MuMO dev n=20"
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_mumo_residual_gate.py" \
      --candidate-manifest "$PLANNER_MANIFEST" \
      --summary-csv "$EVAL_DIR/external_multiproperty_summary.csv" \
      --oracle-summary "${ORACLE_CSV%.csv}.summary.json" \
      --baseline-gate "$BASELINE_GATE" \
      --baseline-format-summary "$FORGETTING_DIR/baseline/summary.json" \
      --candidate-format-summary "$FORGETTING_DIR/residual/summary.json" \
      --baseline-preference-summary "$PREFERENCE_EVAL_DIR/baseline.json" \
      --candidate-preference-summary "$PREFERENCE_EVAL_DIR/residual.json" \
      --preference-manifest "$PREFERENCE_DIR/manifest.json" \
      --training-summary "$MODEL_DIR/training_summary.json" \
      --output-dir "$GATE_DIR"
    ;;
  *) echo "ERROR: unknown stage $STAGE" >&2; exit 2 ;;
esac
