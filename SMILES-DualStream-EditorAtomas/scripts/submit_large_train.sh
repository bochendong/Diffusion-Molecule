#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"

CONFIG="${SDEA_CONFIG:-$PROJECT_DIR/configs/large.yaml}"
PYTHON_BIN="${SDEA_PYTHON_BIN:-python3}"
LOG_DIR="${SDEA_LOG_DIR:-$PROJECT_DIR/logs}"
RUN_TESTS="${SDEA_RUN_TESTS:-1}"
DRY_RUN="${SDEA_DRY_RUN:-0}"
LOCAL_RUN="${SDEA_LOCAL_RUN:-0}"
REQUIRE_CUDA="${SDEA_REQUIRE_CUDA:-1}"
GPU_PROFILE="${SDEA_GPU_PROFILE:-h100_20gb_mig}"
RESOURCE_PROFILE="${SDEA_RESOURCE_PROFILE:-auto}"

SLURM_ACCOUNT="${SDEA_SLURM_ACCOUNT:-def-hup-ab}"
SLURM_JOB_NAME="${SDEA_SLURM_JOB_NAME:-sdea-large}"

mkdir -p "$LOG_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: SDEA_PYTHON_BIN is not executable or on PATH: $PYTHON_BIN" >&2
  echo "       Set SDEA_PYTHON_BIN to a Python with torch for training." >&2
  exit 2
fi

if [[ "$RESOURCE_PROFILE" == "auto" ]]; then
  DIRECT_GPU_REQUEST="${SDEA_SLURM_GPUS:-}"
  if [[ "$GPU_PROFILE" == "h100_40gb_mig" || "$GPU_PROFILE" == "h100_full" || "$DIRECT_GPU_REQUEST" == *"40gb"* || "$DIRECT_GPU_REQUEST" == "h100:1" ]]; then
    RESOURCE_PROFILE="throughput_40gb"
  else
    RESOURCE_PROFILE="economy"
  fi
fi

case "$RESOURCE_PROFILE" in
  economy)
    DEFAULT_GPUS="h100_2g.20gb:1"
    DEFAULT_CPUS="4"
    DEFAULT_MEM="32G"
    DEFAULT_TIME="24:00:00"
    export SDEA_BATCH_SIZE="${SDEA_BATCH_SIZE:-256}"
    export SDEA_EVAL_BATCH_SIZE="${SDEA_EVAL_BATCH_SIZE:-512}"
    export SDEA_EMBED_DIM="${SDEA_EMBED_DIM:-256}"
    export SDEA_HIDDEN_DIM="${SDEA_HIDDEN_DIM:-512}"
    ;;
  throughput_40gb)
    DEFAULT_GPUS="h100_3g.40gb:1"
    DEFAULT_CPUS="8"
    DEFAULT_MEM="64G"
    DEFAULT_TIME="48:00:00"
    export SDEA_BATCH_SIZE="${SDEA_BATCH_SIZE:-512}"
    export SDEA_EVAL_BATCH_SIZE="${SDEA_EVAL_BATCH_SIZE:-1024}"
    export SDEA_EMBED_DIM="${SDEA_EMBED_DIM:-384}"
    export SDEA_HIDDEN_DIM="${SDEA_HIDDEN_DIM:-768}"
    ;;
  *)
    echo "ERROR: unsupported SDEA_RESOURCE_PROFILE=$RESOURCE_PROFILE" >&2
    echo "       Use economy, throughput_40gb, or auto." >&2
    exit 2
    ;;
esac

SLURM_GPUS="${SDEA_SLURM_GPUS:-$DEFAULT_GPUS}"
SLURM_CPUS="${SDEA_SLURM_CPUS:-$DEFAULT_CPUS}"
SLURM_MEM="${SDEA_SLURM_MEM:-$DEFAULT_MEM}"
SLURM_TIME="${SDEA_SLURM_TIME:-$DEFAULT_TIME}"

echo "SMILES DualStream Editor-Atomas large training"
echo "  repo=$REPO_ROOT"
echo "  config=$CONFIG"
echo "  python=$PYTHON_BIN"
echo "  resource_profile=$RESOURCE_PROFILE"
echo "  gpu_profile=$GPU_PROFILE"
echo "  slurm_gpus=$SLURM_GPUS"
echo "  batch_size=$SDEA_BATCH_SIZE"
echo "  eval_batch_size=$SDEA_EVAL_BATCH_SIZE"
echo "  embed_dim=$SDEA_EMBED_DIM"
echo "  hidden_dim=$SDEA_HIDDEN_DIM"
echo "  require_cuda=$REQUIRE_CUDA"
echo "  local_run=$LOCAL_RUN dry_run=$DRY_RUN"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/preflight_large.py" --config "$CONFIG"

if [[ "$DRY_RUN" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_large_pipeline.py" --config "$CONFIG" --dry-run
  exit 0
fi

if [[ "$LOCAL_RUN" == "1" ]]; then
  cd "$REPO_ROOT"
  preflight_args=(--config "$CONFIG" --require-torch)
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    preflight_args+=(--require-cuda)
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/preflight_large.py" "${preflight_args[@]}"
  if [[ "$RUN_TESTS" == "1" ]]; then
    "$PYTHON_BIN" -m unittest discover -s "$PROJECT_DIR/tests" -q
  fi
  exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_large_pipeline.py" --config "$CONFIG"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Set SDEA_LOCAL_RUN=1 to run in this shell." >&2
  exit 1
fi

REQUIRE_CUDA_ARG=""
if [[ "$REQUIRE_CUDA" == "1" ]]; then
  REQUIRE_CUDA_ARG="--require-cuda"
fi

sbatch \
  --account="$SLURM_ACCOUNT" \
  --gpus="$SLURM_GPUS" \
  --cpus-per-task="$SLURM_CPUS" \
  --mem="$SLURM_MEM" \
  --time="$SLURM_TIME" \
  --job-name="$SLURM_JOB_NAME" \
  --output="$LOG_DIR/%x-%j.out" \
  --export=ALL \
  --wrap "cd '$REPO_ROOT' && '$PYTHON_BIN' '$PROJECT_DIR/scripts/preflight_large.py' --config '$CONFIG' --require-torch $REQUIRE_CUDA_ARG && if [ '$RUN_TESTS' = '1' ]; then '$PYTHON_BIN' -m unittest discover -s '$PROJECT_DIR/tests' -q; fi && '$PYTHON_BIN' '$PROJECT_DIR/scripts/run_large_pipeline.py' --config '$CONFIG'"
