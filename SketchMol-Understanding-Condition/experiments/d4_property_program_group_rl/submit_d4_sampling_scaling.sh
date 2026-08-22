#!/usr/bin/env bash
# Submit the four D4 n=256 pools in parallel, then a dependent CPU finalizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found; run this launcher on Nibi" >&2; exit 2; }

ACCOUNT="${SUCC_D4_ACCOUNT:-def-hup-ab}"
MAIL_USER="${SUCC_D4_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SUCC_D4_LOG_DIR:-$PROJECT_DIR/logs/d4_property_program_group_rl}"
mkdir -p "$LOG_DIR"

VARIANTS=(
  two_p_to_seven_p_sft
  two_p_to_seven_p_group_rl
  ood_sft
  ood_group_rl
)
job_ids=()

for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == two_p_to_seven_p_* ]]; then
    time_limit="${SUCC_D4_2P7P_TIME:-20:00:00}"
    short_name="${variant/two_p_to_seven_p_/2p7p-}"
  else
    time_limit="${SUCC_D4_OOD_TIME:-06:00:00}"
    short_name="$variant"
  fi
  sbatch_args=(
    --parsable
    --account="$ACCOUNT"
    --job-name="d4-$short_name"
    --time="$time_limit"
    --mem="${SUCC_D4_MEM:-24G}"
    --cpus-per-task="${SUCC_D4_CPUS:-4}"
    --output="$LOG_DIR/d4-$short_name-%j.log"
    --mail-user="$MAIL_USER"
    --mail-type=END,FAIL
    --export="ALL,D4_VARIANT=$variant"
  )
  job_id=""
  for gpu in \
    "${SUCC_D4_GPU_PRIMARY:-nvidia_h100_80gb_hbm3_2g.20gb:1}" \
    "${SUCC_D4_GPU_FALLBACK:-nvidia_h100_80gb_hbm3_3g.40gb:1}"; do
    if output="$(sbatch "${sbatch_args[@]}" --gres="gpu:$gpu" --wrap="bash '$SCRIPT_DIR/run_d4_candidate_pool.sh'")"; then
      job_id="${output%%;*}"
      break
    fi
  done
  if [[ -z "$job_id" ]]; then
    echo "ERROR: failed to submit $variant" >&2
    exit 1
  fi
  job_ids+=("$job_id")
  echo "$variant=$job_id"
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
final_job="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="d4-finalize" \
  --time="${SUCC_D4_FINALIZE_TIME:-01:30:00}" \
  --mem="${SUCC_D4_FINALIZE_MEM:-16G}" \
  --cpus-per-task="${SUCC_D4_FINALIZE_CPUS:-4}" \
  --dependency="afterok:$dependency" \
  --output="$LOG_DIR/d4-finalize-%j.log" \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/finalize_d4_sampling_scaling.sh'")"
final_job="${final_job%%;*}"

echo "d4_finalize=$final_job"
echo "d4_dependency=$dependency"
echo "d4_log_dir=$LOG_DIR"
echo "d4_report=${SUCC_D4_OUTPUT_ROOT:-$PROJECT_DIR/outputs/d4_property_program_group_rl_seed7}/final/d4_report.md"
