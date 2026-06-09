#!/usr/bin/env bash
# Run a small Unified 3M MolEdit-Instruct pipeline smoke test.

set -euo pipefail

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SMU3M_DATASET_MODE=moledit
export SMU3M_OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_smoke}"
export SMU3M_MOLEDIT_TRAIN_SPLIT="${SMU3M_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SMU3M_MOLEDIT_EVAL_SPLIT="${SMU3M_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SMU3M_MIN_EDIT_SOURCE_TANIMOTO="${SMU3M_MIN_EDIT_SOURCE_TANIMOTO:-0.0}"
export SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS="${SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS:-0}"
export SMU3M_REQUIRE_EVAL_ORACLE_STRICT="${SMU3M_REQUIRE_EVAL_ORACLE_STRICT:-0}"
export SMU3M_DESCRIPTION_LIMIT="${SMU3M_DESCRIPTION_LIMIT:-200}"
export SMU3M_EDIT_LIMIT="${SMU3M_EDIT_LIMIT:-500}"
export SMU3M_MOLEDIT_TRAIN_LIMIT="${SMU3M_MOLEDIT_TRAIN_LIMIT:-500}"
export SMU3M_MOLEDIT_EVAL_LIMIT="${SMU3M_MOLEDIT_EVAL_LIMIT:-1000}"
export SMU3M_TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-500}"
export SMU3M_EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-1000}"

bash "$(dirname "${BASH_SOURCE[0]}")/run_unified_generation_smoke.sh"
