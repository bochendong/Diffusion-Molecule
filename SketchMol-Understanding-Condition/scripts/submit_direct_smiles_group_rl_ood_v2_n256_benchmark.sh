#!/usr/bin/env bash
# Submit the OOD group-RL n=256 benchmark-only run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUCC_DIRECT_OOD_GROUP_RL_SLURM_JOB_NAME="${SUCC_DIRECT_OOD_GROUP_RL_SLURM_JOB_NAME:-succ-direct-smiles-ood-v2-group-rl-n256}"

bash "$SCRIPT_DIR/submit_direct_smiles_group_rl_ood_v2.sh"
