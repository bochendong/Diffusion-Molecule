#!/usr/bin/env bash
# Submit UMTP v1 train -> train-only search distillation -> formal task evaluation -> collection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_PYTHON_BIN="$PYTHON_BIN"
# Nibi requires the allocation name (not the legacy *_gpu association name).
# Use the RAC allocation by default; select RAS with UMTP_SLURM_ACCOUNT=def-hup-ab.
ACCOUNT="${UMTP_SLURM_ACCOUNT:-rrg-hup}"
GPU="${UMTP_SLURM_GPUS:-h100:1}"
MEM="${UMTP_SLURM_MEM:-64G}"
CPUS="${UMTP_SLURM_CPUS:-8}"
PARTITION="${UMTP_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
TRAIN_TIME="${UMTP_TRAIN_SLURM_TIME:-24:00:00}"
DISTILL_TIME="${UMTP_DISTILL_SLURM_TIME:-12:00:00}"
EVAL_TIME="${UMTP_EVAL_SLURM_TIME:-24:00:00}"
MAIL_USER="${UMTP_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_v1}"
TRAIN_SEEDS_RAW="${UMTP_TRAIN_SEEDS:-7}"
EVAL_SEEDS_RAW="${UMTP_EVAL_SEEDS:-101}"
TASKS_RAW="${UMTP_EVAL_TASKS:-2p7p,ood,table1}"

IFS=',' read -r -a TRAIN_SEEDS <<< "$TRAIN_SEEDS_RAW"
IFS=',' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_RAW"
IFS=',' read -r -a TASKS <<< "$TASKS_RAW"
mkdir -p "$LOG_DIR"

common=(--account="$ACCOUNT" --mem="$MEM" --cpus-per-task="$CPUS" --gpus="$GPU")
[[ -n "$PARTITION" ]] && common+=(--partition="$PARTITION")
failure_mail=()
begin_mail=()
end_mail=()
if [[ -n "$MAIL_USER" ]]; then
  failure_mail=(--mail-user="$MAIL_USER" --mail-type=FAIL)
  begin_mail=(--mail-user="$MAIL_USER" --mail-type=BEGIN,FAIL)
  end_mail=(--mail-user="$MAIL_USER" --mail-type=END,FAIL)
fi

eval_ids=()
begin_mail_pending=1
for train_seed in "${TRAIN_SEEDS[@]}"; do
  train_name="umtp-train-s${train_seed}"
  train_mail=("${failure_mail[@]}")
  if ((begin_mail_pending)); then
    train_mail=("${begin_mail[@]}")
    begin_mail_pending=0
  fi
  train_output="$(sbatch --parsable "${common[@]}" "${train_mail[@]}" --time="$TRAIN_TIME" --job-name="$train_name" \
    --output="$LOG_DIR/${train_name}-%j.log" \
    --export="ALL,UMTP_TRAIN_SEED=$train_seed" \
    --wrap="bash '$SCRIPT_DIR/run_umtp_v1_train.sh'")"
  train_id="${train_output%%;*}"

  distill_name="umtp-distill-s${train_seed}"
  distill_output="$(sbatch --parsable "${common[@]}" "${failure_mail[@]}" --time="$DISTILL_TIME" --job-name="$distill_name" \
    --dependency="afterok:$train_id" \
    --output="$LOG_DIR/${distill_name}-%j.log" \
    --export="ALL,UMTP_TRAIN_SEED=$train_seed" \
    --wrap="bash '$SCRIPT_DIR/run_umtp_v1_search_distill.sh'")"
  distill_id="${distill_output%%;*}"
  echo "train_seed=$train_seed train_job_id=$train_id distill_job_id=$distill_id"

  for eval_seed in "${EVAL_SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      eval_name="umtp-${train_seed}-${eval_seed}-${task}"
      eval_output="$(sbatch --parsable "${common[@]}" "${failure_mail[@]}" --time="$EVAL_TIME" --job-name="$eval_name" \
        --dependency="afterok:$distill_id" \
        --output="$LOG_DIR/${eval_name}-%j.log" \
        --export="ALL,UMTP_TRAIN_SEED=$train_seed,UMTP_EVAL_SEED=$eval_seed,UMTP_EVAL_TASK=$task" \
        --wrap="bash '$SCRIPT_DIR/run_umtp_v1_eval_one.sh'")"
      eval_id="${eval_output%%;*}"
      eval_ids+=("$eval_id")
      echo "train_seed=$train_seed eval_seed=$eval_seed task=$task eval_job_id=$eval_id"
    done
  done
done

dependency="$(IFS=:; echo "${eval_ids[*]}")"
collect_cmd="'$PYTHON_BIN' '$SCRIPT_DIR/collect_umtp_v1_results.py' --eval-root '$POLICY_ROOT/eval' --output-prefix '$POLICY_ROOT/eval/umtp_v1'"
collect_output="$(sbatch --parsable --account="$ACCOUNT" "${end_mail[@]}" --time=01:00:00 --mem=8G --cpus-per-task=2 \
  --job-name=umtp-collect --dependency="afterok:$dependency" \
  --output="$LOG_DIR/umtp-collect-%j.log" --wrap="$collect_cmd")"

echo "UMTP v1 pipeline submitted: train_seeds=$TRAIN_SEEDS_RAW eval_seeds=$EVAL_SEEDS_RAW tasks=$TASKS_RAW"
echo "collector_job_id=${collect_output%%;*} eval_jobs=${#eval_ids[@]}"
