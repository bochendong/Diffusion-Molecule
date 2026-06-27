#!/usr/bin/env bash
# Submit the OOD group-RL conservative-decoding benchmark at n=256.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUCC_DIRECT_OOD_GROUP_RL_SLURM_JOB_NAME="${SUCC_DIRECT_OOD_GROUP_RL_SLURM_JOB_NAME:-succ-direct-smiles-ood-v2-group-rl-conservative-n256}"
export SUCC_DIRECT_OOD_GROUP_RL_SLURM_TIME="${SUCC_DIRECT_OOD_GROUP_RL_SLURM_TIME:-08:00:00}"
export SUCC_DIRECT_OOD_GROUP_RL_RUN_SCRIPT="${SUCC_DIRECT_OOD_GROUP_RL_RUN_SCRIPT:-$SCRIPT_DIR/run_direct_smiles_group_rl_ood_v2_conservative_n256_benchmark.sh}"

bash "$SCRIPT_DIR/submit_direct_smiles_group_rl_ood_v2.sh"
