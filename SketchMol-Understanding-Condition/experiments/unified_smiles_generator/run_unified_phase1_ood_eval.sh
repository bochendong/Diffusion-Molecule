#!/usr/bin/env bash
# Phase-1 fixed direct_compat eval on de novo OOD rows (full eval set).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

OOD_ROOT="${SUCC_UNIFIED_OOD_DIRECT_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_mixed_condition}"
OUTPUT_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_phase1_ood_direct_compat_v1}"

export SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT="$OUTPUT_ROOT"
export SUCC_UNIFIED_DIRECT_WARMSTART_RUN_GRPO=0
export SUCC_UNIFIED_DIRECT_WARMSTART_RUN_BASELINE=1
export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT=direct_compat
export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_FEATURE_VARIANT=full
export SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY=with_image
export SUCC_UNIFIED_DIRECT_WARMSTART_FILTER_BENCHMARK_CONTAINS=ood
export SUCC_UNIFIED_DIRECT_WARMSTART_BENCHMARK_TASKS=denovo_ood
export SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR="$OOD_ROOT/train_condition_features_hf_vlm"
export SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR="$OOD_ROOT/eval_condition_features_hf_vlm"
unset SUCC_UNIFIED_EVAL_LIMIT || true

export SUCC_UNIFIED_DECODING_MODE=sample
export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-128}"
export SUCC_UNIFIED_TEMPERATURE="${SUCC_UNIFIED_TEMPERATURE:-0.85}"
export SUCC_UNIFIED_TOP_K="${SUCC_UNIFIED_TOP_K:-40}"
export SUCC_UNIFIED_TOP_P="${SUCC_UNIFIED_TOP_P:-0.95}"
export SUCC_UNIFIED_PARALLEL_SAMPLES="${SUCC_UNIFIED_PARALLEL_SAMPLES:-8}"
export SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES="${SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES:-128}"

echo "Unified phase1 OOD direct_compat eval"
echo "  output_root=$OUTPUT_ROOT"
echo "  ood_root=$OOD_ROOT"
echo "  decoding_mode=$SUCC_UNIFIED_DECODING_MODE"
echo "  num_samples=$SUCC_UNIFIED_NUM_SAMPLES"
echo "  eval_limit=${SUCC_UNIFIED_EVAL_LIMIT:-all}"
echo "  benchmark_tasks=$SUCC_UNIFIED_DIRECT_WARMSTART_BENCHMARK_TASKS"

bash "$SCRIPT_DIR/run_unified_smiles_generator_direct_warmstart_grpo_beam.sh"
