#!/usr/bin/env bash
# Submit the SketchMol paper-reproduction benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ACCOUNT="${SKETCHMOL_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHMOL_SLURM_TIME:-08:00:00}"
MEM="${SKETCHMOL_SLURM_MEM:-32G}"
CPUS="${SKETCHMOL_SLURM_CPUS:-4}"
GPU_PROFILE="${SKETCHMOL_GPU_PROFILE:-h100_40gb_mig}"
JOB_NAME="${SKETCHMOL_SLURM_JOB_NAME:-sketchmol-paper-repro}"
LOG_DIR="${SKETCHMOL_LOG_DIR:-SketchMolBenchmark/logs}"
mkdir -p "$LOG_DIR"

export SKETCHMOL_REPO="${SKETCHMOL_REPO:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
export SKETCHMOL_PYTHON_BIN="${SKETCHMOL_PYTHON_BIN:-python}"
export SKETCHMOL_RUN_NAME="${SKETCHMOL_RUN_NAME:-paper_repro_mw400_$(date +%Y%m%d_%H%M%S)}"

require_existing_file() {
  local env_name="$1"
  local description="$2"
  local example="$3"
  local value="${!env_name:-}"

  if [[ -z "$value" ]]; then
    echo "ERROR: set $env_name=$example" >&2
    exit 2
  fi
  if [[ "$value" == "/path/to/"* ]]; then
    echo "ERROR: $env_name is still the example placeholder: $value" >&2
    echo "       Replace it with the real $description path." >&2
    exit 2
  fi
  if [[ ! -f "$value" ]]; then
    echo "ERROR: $env_name must point to a real $description file: $value" >&2
    exit 2
  fi
  if [[ ! -r "$value" ]]; then
    echo "ERROR: $env_name exists but is not readable: $value" >&2
    exit 2
  fi
}

require_executable() {
  local env_name="$1"
  local example="$2"
  local value="${!env_name:-}"

  if [[ -z "$value" ]]; then
    echo "ERROR: set $env_name=$example" >&2
    exit 2
  fi
  if ! command -v "$value" >/dev/null 2>&1 && [[ ! -x "$value" ]]; then
    echo "ERROR: $env_name is not executable: $value" >&2
    exit 2
  fi
}

if [[ ! -d "$SKETCHMOL_REPO" ]]; then
  echo "ERROR: real SketchMol repo not found: $SKETCHMOL_REPO" >&2
  exit 2
fi

require_executable "SKETCHMOL_PYTHON_BIN" "/path/to/sketchmol/python"
require_executable "SKETCHMOL_MOLSCRIBE_PYTHON_BIN" "/path/to/molscribe/python"
require_existing_file \
  "SKETCHMOL_CKPT" \
  "SketchMol diffusion checkpoint" \
  "/absolute/path/to/sketchmol/model.ckpt"
require_existing_file \
  "SKETCHMOL_MOLSCRIBE_MODEL" \
  "MolScribe checkpoint" \
  "/absolute/path/to/swin_base_char_aux_200k.pth"

if [[ ! -d "${SKETCHMOL_MOLSCRIBE_WORKDIR:-$SKETCHMOL_REPO/evaluate}" ]]; then
  echo "ERROR: SKETCHMOL_MOLSCRIBE_WORKDIR does not exist: ${SKETCHMOL_MOLSCRIBE_WORKDIR:-$SKETCHMOL_REPO/evaluate}" >&2
  echo "       For paper reproduction, prefer a legacy MolScribe checkout with SketchMol's predict_csv.py copied in." >&2
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

if [[ -n "${SKETCHMOL_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SKETCHMOL_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1" "h100_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("h100_2g.20gb:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("h100_3g.40gb:1" "nvidia_h100_80gb_hbm3_3g.40gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100_80gb:1" "h100:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

echo "Submitting SketchMol paper reproduction benchmark:"
echo "  run_name=$SKETCHMOL_RUN_NAME"
echo "  sketchmol_repo=$SKETCHMOL_REPO"
echo "  sketchmol_python=$SKETCHMOL_PYTHON_BIN"
echo "  molscribe_python=$SKETCHMOL_MOLSCRIBE_PYTHON_BIN"
echo "  molscribe_workdir=${SKETCHMOL_MOLSCRIBE_WORKDIR:-$SKETCHMOL_REPO/evaluate}"
echo "  ckpt=$SKETCHMOL_CKPT"
echo "  molscribe_model=$SKETCHMOL_MOLSCRIBE_MODEL"
echo "  preset=${SKETCHMOL_PRESET_STR:-MW:400}"
echo "  scale=${SKETCHMOL_SCALE:-1.2}"
echo "  scale_pro=${SKETCHMOL_SCALE_PRO:-6.3}"
echo "  conditional_count=${SKETCHMOL_CONDITIONAL_COUNT:-40}"
echo "  gpu_candidates=${GPU_CANDIDATES[*]}"
echo "  slurm_time=$TIME"
echo "  slurm_mem=$MEM"
echo "  slurm_cpus=$CPUS"

SUBMITTED=0
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  echo "Trying sbatch with --gpus=$GPU_REQUEST"
  if sbatch \
    --account="$ACCOUNT" \
    --job-name="$JOB_NAME" \
    --time="$TIME" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --gpus="$GPU_REQUEST" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="bash 'SketchMolBenchmark/scripts/run_paper_repro.sh'"; then
    SUBMITTED=1
    break
  fi
done

if [[ "$SUBMITTED" != "1" ]]; then
  echo "ERROR: failed to submit with GPU candidates: ${GPU_CANDIDATES[*]}" >&2
  exit 1
fi
