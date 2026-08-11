#!/usr/bin/env bash
# Train and gate the first closed-loop common-LLM GraphEditDSL policy pilot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
SFT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
SFT_DATA="${SUCC_UCA_SFT_DATA:-$SFT_ROOT/data/common_llm_sft}"
INPUT_ADAPTER="${SUCC_UCA_INPUT_ADAPTER:-$SFT_ROOT/model/seed_1703/adapter}"
SEED="${SUCC_UCA_SEED:-1707}"
RUN_ROOT="${SUCC_UCA_TOOL_POLICY_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_tool_policy_grpo_v1/seed_${SEED}}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SUCC_GSK3B_ORACLE_PATH="$GSK3B_ORACLE"
export SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN"
export SUCC_ADMET_PYTHONPATH="${SUCC_ADMET_PYTHONPATH:-/cvmfs/soft.computecanada.ca/easybuild/python/site-packages:/cvmfs/soft.computecanada.ca/custom/python/site-packages}"

if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi

for path in \
  "$PYTHON_BIN" \
  "$ADMET_PYTHON_BIN" \
  "$SFT_DATA/train.jsonl" \
  "$SFT_DATA/validation.jsonl" \
  "$INPUT_ADAPTER/adapter_model.safetensors" \
  "$GSK3B_ORACLE"; do
  [[ -e "$path" ]] || { echo "ERROR: missing tool-policy input: $path" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT"

echo "=== Verify typed executor and pinned benchmark oracle ==="
"$PYTHON_BIN" "$CODE_PROJECT_DIR/experiments/unified_smiles_generator/legacy_gsk3b_oracle.py" \
  verify --model "$GSK3B_ORACLE"
"$ADMET_PYTHON_BIN" -c "import numpy; from admet_ai import ADMETModel; print('ADMET-AI bridge dependencies verified')"

echo "=== Train closed-loop common-LLM tool policy on train conditions ==="
"$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_tool_policy_grpo.py" \
  --train-jsonl "$SFT_DATA/train.jsonl" \
  --validation-jsonl "$SFT_DATA/validation.jsonl" \
  --input-adapter-dir "$INPUT_ADAPTER" \
  --output-dir "$RUN_ROOT" \
  --base-model "$BASE_MODEL" \
  --rollout-origins "${SUCC_UCA_ROLLOUT_ORIGINS:-table1,mumo}" \
  --max-train-per-origin "${SUCC_UCA_MAX_TRAIN_PER_ORIGIN:-8}" \
  --max-validation-per-origin "${SUCC_UCA_MAX_VALIDATION_PER_ORIGIN:-4}" \
  --rollouts-per-condition "${SUCC_UCA_ROLLOUTS_PER_CONDITION:-4}" \
  --validation-rollouts "${SUCC_UCA_VALIDATION_ROLLOUTS:-6}" \
  --max-steps "${SUCC_UCA_MAX_STEPS:-2}" \
  --site-limit "${SUCC_UCA_SITE_LIMIT:-24}" \
  --max-actions "${SUCC_UCA_MAX_ACTIONS:-16}" \
  --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}" \
  --max-length "${SUCC_UCA_MAX_LENGTH:-768}" \
  --temperature "${SUCC_UCA_TEMPERATURE:-0.8}" \
  --epochs "${SUCC_UCA_EPOCHS:-1}" \
  --gradient-accumulation "${SUCC_UCA_GRADIENT_ACCUMULATION:-4}" \
  --learning-rate "${SUCC_UCA_LR:-3e-6}" \
  --anchor-sft-weight "${SUCC_UCA_ANCHOR_SFT_WEIGHT:-0.10}" \
  --anchor-max-per-origin "${SUCC_UCA_ANCHOR_MAX_PER_ORIGIN:-64}" \
  --seed "$SEED"

echo "Common-LLM tool-policy pilot ready: $RUN_ROOT"
echo "Gate: $RUN_ROOT/gate.json"
