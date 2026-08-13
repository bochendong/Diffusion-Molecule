#!/usr/bin/env bash
# Train and evaluate the 200-condition event-driven feedback-agent signal.

set -euo pipefail

STAGE="${1:?usage: run_feedback_repair_v14a.sh prepare|gpu|gpu_generate|deterministic|merge_llm|merge_deterministic|oracle_llm|oracle_deterministic|gate}"
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
EVIDENCE_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711"
V8_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711"
V12_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_direct_repair_v12/seed_1715"
SFT_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1/data/common_llm_sft"
RUN_ROOT="${SUCC_UCA_FEEDBACK_REPAIR_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_feedback_repair_v14a/seed_1716}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="$RUN_ROOT/controller/data"
MODEL_DIR="$RUN_ROOT/controller/model"
VALIDATION_DIR="$RUN_ROOT/controller/validation"
LLM_DIR="$RUN_ROOT/llm"
DET_DIR="$RUN_ROOT/deterministic"
DEV_SOURCES="$V8_ROOT/data/dev_sources.jsonl"
FEEDBACK_REPAIR_SHARD_COUNT="${SUCC_UCA_FEEDBACK_REPAIR_SHARD_COUNT:-8}"
SCORE_BATCH_SIZE="${SUCC_UCA_FEEDBACK_SCORE_BATCH_SIZE:-4}"

export PYTHONPATH="$DEP_OVERLAY:$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$DATA_DIR" "$MODEL_DIR" "$VALIDATION_DIR/baseline" "$VALIDATION_DIR/candidate" \
  "$LLM_DIR/oracle" "$LLM_DIR/evaluation" "$DET_DIR/oracle" "$DET_DIR/evaluation" "$RUN_ROOT/gate"

evaluate_candidates() {
  local candidate_dir="$1"
  local title="$2"
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$candidate_dir/exact_n20.csv" \
    --output-dir "$candidate_dir/evaluation" \
    --generated-properties-csv "$candidate_dir/oracle/generated_properties.csv" \
    --source-properties-csv "$candidate_dir/oracle/generated_properties.csv" \
    --group-column condition_id --min-source-tanimoto 0.4 --report-title "$title"
}

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" "$SCRIPT_DIR/build_feedback_repair_controller_sft.py" \
      --data-dir "$EVIDENCE_ROOT/data" --evidence-root "$EVIDENCE_ROOT" \
      --stable-sft-dir "$SFT_ROOT" --output-dir "$DATA_DIR" \
      --max-states-per-task 384 --replay-per-origin 256 \
      --max-committed-edits 3 --max-proposals 6 --seed 1716
    ;;
  gpu)
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_feedback_repair_controller.py" \
      --feedback-validation-jsonl "$DATA_DIR/validation.jsonl" \
      --common-validation-jsonl "$SFT_ROOT/validation.jsonl" \
      --adapter-dir "$V12_ROOT/controller/model/adapter" \
      --output-json "$VALIDATION_DIR/baseline/summary.json" --base-model "$BASE_MODEL"
    "$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_lora.py" \
      --train-jsonl "$DATA_DIR/train.jsonl" --validation-jsonl "$DATA_DIR/validation.jsonl" \
      --input-adapter-dir "$V12_ROOT/controller/model/adapter" --output-dir "$MODEL_DIR" \
      --base-model "$BASE_MODEL" --max-length 512 --epochs 1 --batch-size 2 \
      --gradient-accumulation 8 --learning-rate 2e-6 --compute-dtype float32 --seed 1716
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_feedback_repair_controller.py" \
      --feedback-validation-jsonl "$DATA_DIR/validation.jsonl" \
      --common-validation-jsonl "$SFT_ROOT/validation.jsonl" \
      --adapter-dir "$MODEL_DIR/adapter" \
      --output-json "$VALIDATION_DIR/candidate/summary.json" --base-model "$BASE_MODEL"
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_feedback_repair_trajectories.py" \
      --evidence-root "$EVIDENCE_ROOT" --dev-sources-jsonl "$DEV_SOURCES" \
      --output-csv "$LLM_DIR/exact_n20.csv" --manifest-json "$LLM_DIR/manifest.json" \
      --controller-mode llm --adapter-dir "$MODEL_DIR/adapter" --base-model "$BASE_MODEL" \
      --conditions-per-split 100 --attempts-per-condition 20 \
      --max-committed-edits 3 --max-proposals 6 --max-symbolic-actions 256 \
      --action-retry-limit 12 --temperature 0.75 --min-source-tanimoto 0.4 --seed 1716
    ;;
  gpu_generate)
    [[ -s "$MODEL_DIR/adapter/adapter_model.safetensors" \
      && -s "$MODEL_DIR/training_summary.json" \
      && -s "$VALIDATION_DIR/baseline/summary.json" \
      && -s "$VALIDATION_DIR/candidate/summary.json" ]] || {
      echo "ERROR: the completed v14a adapter and validation artifacts are unavailable" >&2; exit 2;
    }
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
    SHARD_TAG="$(printf '%03d' "$SHARD_INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_feedback_repair_trajectories.py" \
      --evidence-root "$EVIDENCE_ROOT" --dev-sources-jsonl "$DEV_SOURCES" \
      --output-csv "$LLM_DIR/shards/trajectories_${SHARD_TAG}.csv" \
      --manifest-json "$LLM_DIR/shards/manifest_${SHARD_TAG}.json" \
      --controller-mode llm --adapter-dir "$MODEL_DIR/adapter" --base-model "$BASE_MODEL" \
      --conditions-per-split 100 --attempts-per-condition 20 \
      --max-committed-edits 3 --max-proposals 6 --max-symbolic-actions 256 \
      --action-retry-limit 12 --temperature 0.75 --min-source-tanimoto 0.4 \
      --score-batch-size "$SCORE_BATCH_SIZE" --shard-index "$SHARD_INDEX" \
      --shard-count "$FEEDBACK_REPAIR_SHARD_COUNT" --seed 1716
    ;;
  deterministic)
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
    SHARD_TAG="$(printf '%03d' "$SHARD_INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_feedback_repair_trajectories.py" \
      --evidence-root "$EVIDENCE_ROOT" --dev-sources-jsonl "$DEV_SOURCES" \
      --output-csv "$DET_DIR/shards/trajectories_${SHARD_TAG}.csv" \
      --manifest-json "$DET_DIR/shards/manifest_${SHARD_TAG}.json" \
      --controller-mode deterministic --conditions-per-split 100 --attempts-per-condition 20 \
      --max-committed-edits 3 --max-proposals 6 --max-symbolic-actions 256 \
      --action-retry-limit 12 --temperature 0.75 --min-source-tanimoto 0.4 \
      --shard-index "$SHARD_INDEX" --shard-count "$FEEDBACK_REPAIR_SHARD_COUNT" --seed 1716
    ;;
  merge_llm|merge_deterministic)
    if [[ "$STAGE" == "merge_llm" ]]; then
      MERGE_DIR="$LLM_DIR"
      MERGE_MODE="llm"
    else
      MERGE_DIR="$DET_DIR"
      MERGE_MODE="deterministic"
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/merge_feedback_repair_trajectories.py" \
      --shard-dir "$MERGE_DIR/shards" --output-csv "$MERGE_DIR/exact_n20.csv" \
      --manifest-json "$MERGE_DIR/manifest.json" --controller-mode "$MERGE_MODE" \
      --shard-count "$FEEDBACK_REPAIR_SHARD_COUNT" --expected-conditions 200 \
      --expected-ind 100 --expected-ood 100
    ;;
  oracle_llm)
    SUCC_PYTHON_BIN="$PYTHON_BIN" SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$LLM_DIR/exact_n20.csv" \
    SUCC_ORACLE_OUTPUT_CSV="$LLM_DIR/oracle/generated_properties.csv" \
    SUCC_ORACLE_WORK_DIR="$LLM_DIR/oracle/work" \
    SUCC_ORACLE_MERGE_PROPERTIES_CSV="$V12_ROOT/oracle/generated_properties.csv" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES=bbbp,hia,mutagenicity \
    bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    evaluate_candidates "$LLM_DIR" "Common-LLM feedback repair v14a signal exact n=20"
    ;;
  oracle_deterministic)
    SUCC_PYTHON_BIN="$PYTHON_BIN" SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$DET_DIR/exact_n20.csv" \
    SUCC_ORACLE_OUTPUT_CSV="$DET_DIR/oracle/generated_properties.csv" \
    SUCC_ORACLE_WORK_DIR="$DET_DIR/oracle/work" \
    SUCC_ORACLE_MERGE_PROPERTIES_CSV="$V12_ROOT/oracle/generated_properties.csv" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES=bbbp,hia,mutagenicity \
    bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    evaluate_candidates "$DET_DIR" "Deterministic feedback repair v14a signal exact n=20"
    ;;
  gate)
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_feedback_repair_signal.py" \
      --llm-manifest "$LLM_DIR/manifest.json" --deterministic-manifest "$DET_DIR/manifest.json" \
      --llm-summary "$LLM_DIR/evaluation/external_multiproperty_summary.csv" \
      --deterministic-summary "$DET_DIR/evaluation/external_multiproperty_summary.csv" \
      --controller-validation "$VALIDATION_DIR/candidate/summary.json" \
      --baseline-controller-validation "$VALIDATION_DIR/baseline/summary.json" \
      --data-manifest "$DATA_DIR/manifest.json" --training-summary "$MODEL_DIR/training_summary.json" \
      --llm-oracle-summary "$LLM_DIR/oracle/generated_properties.summary.json" \
      --deterministic-oracle-summary "$DET_DIR/oracle/generated_properties.summary.json" \
      --output-dir "$RUN_ROOT/gate"
    ;;
  *) echo "ERROR: unknown stage $STAGE" >&2; exit 2 ;;
esac
