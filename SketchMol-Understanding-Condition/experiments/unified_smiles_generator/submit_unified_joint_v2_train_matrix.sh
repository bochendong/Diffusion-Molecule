#!/usr/bin/env bash
# Submit prepare -> U0 validation -> U1/U2 x three seeds. Each train job selects its checkpoint on validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_UNIFIED_JOINT_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_JOINT_SLURM_TIME:-48:00:00}"
MEM="${SUCC_UNIFIED_JOINT_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_JOINT_SLURM_CPUS:-8}"
GPU="${SUCC_UNIFIED_JOINT_SLURM_GPUS:-h100:1}"
PARTITION="${SUCC_UNIFIED_JOINT_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
SEEDS_RAW="${SUCC_UNIFIED_TRAIN_SEEDS:-7,17,27}"
IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"
mkdir -p "$LOG_DIR"

COMMON=(--account="$ACCOUNT" --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS" --gpus="$GPU")
[[ -n "$PARTITION" ]] && COMMON+=(--partition="$PARTITION")
prepare_id="$(sbatch --parsable "${COMMON[@]}" --job-name=ujv2-prepare --output="$LOG_DIR/ujv2-prepare-%j.log" --export=ALL,SUCC_UNIFIED_JOINT_STAGE=u1,SUCC_UNIFIED_JOINT_PREPARE_ONLY=1 --wrap="bash '$SCRIPT_DIR/run_unified_joint_v2_train.sh'")"
prepare_id="${prepare_id%%;*}"
baseline_id="$(sbatch --parsable "${COMMON[@]}" --job-name=ujv2-u0-val --dependency="afterok:$prepare_id" --output="$LOG_DIR/ujv2-u0-val-%j.log" --export=ALL,SUCC_UNIFIED_JOINT_STAGE=u0 --wrap="bash '$SCRIPT_DIR/run_unified_joint_v2_validation.sh'")"
baseline_id="${baseline_id%%;*}"
echo "prepare_job_id=$prepare_id baseline_validation_job_id=$baseline_id"

train_ids=()
for stage in u1 u2; do
  for seed in "${SEEDS[@]}"; do
    name="ujv2-${stage}-s${seed}"
    output="$(sbatch --parsable "${COMMON[@]}" --job-name="$name" --dependency="afterok:$baseline_id" --output="$LOG_DIR/${name}-%j.log" --export=ALL,SUCC_UNIFIED_JOINT_STAGE="$stage",SUCC_UNIFIED_SEED="$seed",SUCC_UNIFIED_JOINT_SKIP_PREPARE=1 --wrap="bash '$SCRIPT_DIR/run_unified_joint_v2_train.sh'")"
    job_id="${output%%;*}"
    train_ids+=("$job_id")
    echo "stage=$stage train_seed=$seed job_id=$job_id"
  done
done
echo "training_jobs=${#train_ids[@]}"
