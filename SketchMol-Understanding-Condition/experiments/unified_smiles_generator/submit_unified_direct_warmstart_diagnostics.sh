#!/usr/bin/env bash
# Submit warm-start beam diagnostics (no GRPO) for the unified direct-checkpoint line.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_UNIFIED_WARMSTART_DIAG_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_UNIFIED_WARMSTART_DIAG_SLURM_TIME:-4:00:00}"
MEM="${SUCC_UNIFIED_WARMSTART_DIAG_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_WARMSTART_DIAG_SLURM_CPUS:-8}"
GPU_PROFILE="${SUCC_UNIFIED_WARMSTART_DIAG_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_full}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_UNIFIED_WARMSTART_DIAG_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

PROFILE="${SUCC_UNIFIED_WARMSTART_DIAG_PROFILE:-all}"
SEQUENTIAL="${SUCC_UNIFIED_WARMSTART_DIAG_SEQUENTIAL:-1}"
JOB_NAME="${SUCC_UNIFIED_WARMSTART_DIAG_JOB_NAME:-succ-unified-phase0-direct-compat}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi

if [[ -n "${SUCC_UNIFIED_WARMSTART_DIAG_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_UNIFIED_WARMSTART_DIAG_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "none" || "$GPU_PROFILE" == "0" ]]; then
  GPU_CANDIDATES=("")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1" "h100:1" "a100:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

if [[ "$PROFILE" == "all" ]]; then
  PROFILES=(
    direct_compat_with_image
    direct_compat_text_only
    property_program_only
    direct_feature_store
  )
else
  PROFILES=("$PROFILE")
fi

mkdir -p "$LOG_DIR"

submit_one() {
  local profile="$1"
  local job_label="${JOB_NAME}-${profile}"
  local -a SBATCH_ARGS=(
    --account="$ACCOUNT"
    --job-name="$job_label"
    --time="$TIME"
    --mem="$MEM"
    --cpus-per-task="$CPUS"
    --output="$LOG_DIR/${job_label}-%j.log"
    --export=ALL,SUCC_UNIFIED_WARMSTART_DIAG_PROFILE="$profile"
  )
  if [[ -n "$PARTITION" ]]; then
    SBATCH_ARGS+=(--partition="$PARTITION")
  fi

  local job_id=""
  local GPU_REQUEST
  for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
    local output=""
    if [[ -n "$GPU_REQUEST" ]]; then
      if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_unified_direct_warmstart_diagnostics.sh'")"; then
        continue
      fi
    else
      if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$SCRIPT_DIR/run_unified_direct_warmstart_diagnostics.sh'")"; then
        continue
      fi
    fi
    echo "$output"
    job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ -n "$job_id" ]]; then
      break
    fi
  done

  if [[ -z "$job_id" ]]; then
    echo "ERROR: failed to submit profile=$profile" >&2
    return 1
  fi
  echo "warmstart_diag_profile=$profile job_id=$job_id"
}

echo "Submitting unified direct warm-start diagnostics"
echo "  profiles=${PROFILES[*]}"
echo "  sequential=$SEQUENTIAL"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  python=$SUCC_PYTHON_BIN"
echo "  gpu_profile=$GPU_PROFILE"
echo "  eval_limit=${SUCC_UNIFIED_EVAL_LIMIT:-500}"
echo "  run_grpo=${SUCC_UNIFIED_DIRECT_WARMSTART_RUN_GRPO:-0}"

if [[ "$SEQUENTIAL" == "1" && "$PROFILE" == "all" ]]; then
  job_label="${JOB_NAME}-all-seq"
  SBATCH_ARGS=(
    --account="$ACCOUNT"
    --job-name="$job_label"
    --time="$TIME"
    --mem="$MEM"
    --cpus-per-task="$CPUS"
    --output="$LOG_DIR/${job_label}-%j.log"
    --export=ALL,SUCC_UNIFIED_WARMSTART_DIAG_PROFILE=all
  )
  if [[ -n "$PARTITION" ]]; then
    SBATCH_ARGS+=(--partition="$PARTITION")
  fi

  job_id=""
  for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
    output=""
    if [[ -n "$GPU_REQUEST" ]]; then
      echo "Trying sbatch with --gpus=$GPU_REQUEST"
      if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$SCRIPT_DIR/run_unified_direct_warmstart_diagnostics.sh'")"; then
        continue
      fi
    else
      echo "Trying sbatch without GPU request"
      if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$SCRIPT_DIR/run_unified_direct_warmstart_diagnostics.sh'")"; then
        continue
      fi
    fi
    echo "$output"
    job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [[ -n "$job_id" ]]; then
      break
    fi
  done

  if [[ -z "$job_id" ]]; then
    echo "ERROR: failed to submit sequential all-profile diagnostic job." >&2
    exit 1
  fi
  echo "warmstart_diag_mode=sequential_all job_id=$job_id"
  exit 0
fi

for profile in "${PROFILES[@]}"; do
  submit_one "$profile"
done
