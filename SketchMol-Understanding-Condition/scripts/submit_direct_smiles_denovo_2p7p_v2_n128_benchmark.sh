#!/usr/bin/env bash
# Submit direct-SMILES v2 mixed-condition n=128 inference-only benchmark on de novo 2p-7p.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUCC_DIRECT_DENOVO_SLURM_JOB_NAME="${SUCC_DIRECT_DENOVO_SLURM_JOB_NAME:-succ-direct-smiles-2p7p-v2-n128}"
export SUCC_DIRECT_DENOVO_SLURM_TIME="${SUCC_DIRECT_DENOVO_SLURM_TIME:-12:00:00}"
export SUCC_DIRECT_DENOVO_OUTPUT_DIR="${SUCC_DIRECT_DENOVO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition_n128}"

bash "$SCRIPT_DIR/submit_direct_smiles_denovo_2p7p_benchmark.sh"
