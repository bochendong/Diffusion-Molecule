#!/usr/bin/env bash
# Submit direct-SMILES v2 mixed-condition n=128 inference-only benchmark on de novo OOD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUCC_DIRECT_OOD_SLURM_JOB_NAME="${SUCC_DIRECT_OOD_SLURM_JOB_NAME:-succ-direct-smiles-ood-v2-n128}"
export SUCC_DIRECT_OOD_SLURM_TIME="${SUCC_DIRECT_OOD_SLURM_TIME:-06:00:00}"
export SUCC_DIRECT_OOD_OUTPUT_DIR="${SUCC_DIRECT_OOD_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_mixed_condition_n128}"

bash "$SCRIPT_DIR/submit_direct_smiles_denovo_ood_benchmark.sh"
