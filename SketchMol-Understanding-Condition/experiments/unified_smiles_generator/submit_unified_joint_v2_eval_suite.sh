#!/usr/bin/env bash
# Submit the formal Joint v2 matrix as independent task jobs, then one collector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"

ACCOUNT="${SUCC_UNIFIED_JOINT_EVAL_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_JOINT_EVAL_SLURM_TIME:-24:00:00}"
MEM="${SUCC_UNIFIED_JOINT_EVAL_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_JOINT_EVAL_SLURM_CPUS:-8}"
GPU="${SUCC_UNIFIED_JOINT_EVAL_SLURM_GPUS:-h100:1}"
PARTITION="${SUCC_UNIFIED_JOINT_EVAL_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
STAGES_RAW="${SUCC_UNIFIED_JOINT_EVAL_STAGES:-u0,u1,u2}"
TRAIN_SEEDS_RAW="${SUCC_UNIFIED_TRAIN_SEEDS:-7,17,27}"
EVAL_SEEDS_RAW="${SUCC_UNIFIED_EVAL_SEEDS:-101,202,303}"
TASKS_RAW="${SUCC_UNIFIED_JOINT_EVAL_TASKS:-2p7p,ood,table1}"

IFS=',' read -r -a STAGES <<< "$STAGES_RAW"
IFS=',' read -r -a TRAIN_SEEDS <<< "$TRAIN_SEEDS_RAW"
IFS=',' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_RAW"
IFS=',' read -r -a TASKS <<< "$TASKS_RAW"
mkdir -p "$LOG_DIR"

COMMON=(--account="$ACCOUNT" --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS" --gpus="$GPU")
[[ -n "$PARTITION" ]] && COMMON+=(--partition="$PARTITION")
job_ids=()
for stage in "${STAGES[@]}"; do
  if [[ "$stage" == "u0" ]]; then stage_seeds=(base); else stage_seeds=("${TRAIN_SEEDS[@]}"); fi
  for train_seed in "${stage_seeds[@]}"; do
    for eval_seed in "${EVAL_SEEDS[@]}"; do
      for task in "${TASKS[@]}"; do
        name="ujv2-${stage}-${train_seed}-${eval_seed}-${task}"
        export_vars="ALL,SUCC_UNIFIED_JOINT_EVAL_STAGES=$stage,SUCC_UNIFIED_TRAIN_SEEDS=$train_seed,SUCC_UNIFIED_EVAL_SEEDS=$eval_seed,SUCC_UNIFIED_JOINT_EVAL_TASKS=$task,SUCC_UNIFIED_JOINT_SKIP_COLLECT=1"
        output="$(sbatch --parsable "${COMMON[@]}" --job-name="$name" --output="$LOG_DIR/${name}-%j.log" --export="$export_vars" --wrap="bash '$SCRIPT_DIR/run_unified_joint_v2_eval_suite.sh'")"
        job_id="${output%%;*}"
        job_ids+=("$job_id")
        echo "stage=$stage train_seed=$train_seed eval_seed=$eval_seed task=$task job_id=$job_id"
      done
    done
  done
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
JOINT_ROOT="${SUCC_UNIFIED_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
collect_cmd="'$SUCC_PYTHON_BIN' '$SCRIPT_DIR/collect_unified_joint_v2_results.py' --eval-root '$JOINT_ROOT/eval' --selection-root '$JOINT_ROOT' --output-prefix '$JOINT_ROOT/eval/unified_joint_v2'"
collect_output="$(sbatch --parsable --account="$ACCOUNT" --job-name=ujv2-collect --time=01:00:00 --mem=8G --cpus-per-task=2 --dependency="afterok:$dependency" --output="$LOG_DIR/ujv2-collect-%j.log" --wrap="$collect_cmd")"
echo "collector_job_id=${collect_output%%;*} dependency_jobs=${#job_ids[@]}"
