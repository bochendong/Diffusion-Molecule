#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT_DIR/scripts/enhance_moledit_instruct.py"

DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
MOLEDIT_RAW_INPUT="${MOLEDIT_RAW_INPUT:-$DM_DATA_ROOT/raw/moledit-instruct/MolEdit-Instruct_3034459.txt}"
MOLEDIT_OUTPUT_DIR="${MOLEDIT_OUTPUT_DIR:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1}"
MOLEDIT_SHARDS="${MOLEDIT_SHARDS:-64}"
MOLEDIT_PYTHON_BIN="${MOLEDIT_PYTHON_BIN:-python3}"
MOLEDIT_LIMIT="${MOLEDIT_LIMIT:-}"
MOLEDIT_LOCAL="${MOLEDIT_LOCAL:-0}"
MOLEDIT_DRY_RUN="${MOLEDIT_DRY_RUN:-0}"

MOLEDIT_SLURM_ACCOUNT="${MOLEDIT_SLURM_ACCOUNT:-def-hup-ab_gpu}"
MOLEDIT_SLURM_PARTITION="${MOLEDIT_SLURM_PARTITION:-}"
MOLEDIT_JOB_NAME="${MOLEDIT_JOB_NAME:-moledit-enhance}"
MOLEDIT_CPUS_PER_TASK="${MOLEDIT_CPUS_PER_TASK:-4}"
MOLEDIT_MEM="${MOLEDIT_MEM:-24G}"
MOLEDIT_TIME="${MOLEDIT_TIME:-12:00:00}"

MOLEDIT_EVAL_MAX="${MOLEDIT_EVAL_MAX:-50000}"
MOLEDIT_EVAL_PER_BUCKET="${MOLEDIT_EVAL_PER_BUCKET:-2000}"
MOLEDIT_HARD_MAX="${MOLEDIT_HARD_MAX:-50000}"
MOLEDIT_SMOKE_LIMIT="${MOLEDIT_SMOKE_LIMIT:-1000}"
MOLEDIT_FINGERPRINT_BITS="${MOLEDIT_FINGERPRINT_BITS:-2048}"

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      MOLEDIT_DRY_RUN=1
      ;;
    --local)
      MOLEDIT_LOCAL=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$MOLEDIT_SHARDS" -lt 1 ]]; then
  echo "MOLEDIT_SHARDS must be positive" >&2
  exit 2
fi

q() {
  printf "%q" "$1"
}

LIMIT_ARGS=()
if [[ -n "$MOLEDIT_LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$MOLEDIT_LIMIT")
fi

common_args() {
  printf " --output-dir %s --num-shards %s --fingerprint-bits %s" \
    "$(q "$MOLEDIT_OUTPUT_DIR")" \
    "$(q "$MOLEDIT_SHARDS")" \
    "$(q "$MOLEDIT_FINGERPRINT_BITS")"
  if [[ ${#LIMIT_ARGS[@]} -gt 0 ]]; then
    printf " --limit %s" "$(q "$MOLEDIT_LIMIT")"
  fi
}

stage_command() {
  local stage="$1"
  local shard_expr="${2:-}"
  local cmd
  cmd="$(q "$MOLEDIT_PYTHON_BIN") $(q "$SCRIPT") --stage $(q "$stage")"
  case "$stage" in
    normalize-pairs)
      cmd+=" --input $(q "$MOLEDIT_RAW_INPUT")"
      ;;
    finalize)
      cmd+=" --eval-max $(q "$MOLEDIT_EVAL_MAX")"
      cmd+=" --eval-per-bucket $(q "$MOLEDIT_EVAL_PER_BUCKET")"
      cmd+=" --hard-max $(q "$MOLEDIT_HARD_MAX")"
      cmd+=" --smoke-limit $(q "$MOLEDIT_SMOKE_LIMIT")"
      ;;
  esac
  cmd+="$(common_args)"
  if [[ -n "$shard_expr" ]]; then
    cmd+=" --shard-index $shard_expr"
  fi
  printf "%s" "$cmd"
}

print_config() {
  cat <<EOF
MolEdit-Instruct enhancement jobs
raw input:        $MOLEDIT_RAW_INPUT
output dir:       $MOLEDIT_OUTPUT_DIR
shards:           $MOLEDIT_SHARDS
python:           $MOLEDIT_PYTHON_BIN
fingerprint bits: $MOLEDIT_FINGERPRINT_BITS
limit:            ${MOLEDIT_LIMIT:-<none>}
local:            $MOLEDIT_LOCAL
dry run:          $MOLEDIT_DRY_RUN
EOF
}

run_local() {
  print_config
  for i in $(seq 0 $((MOLEDIT_SHARDS - 1))); do
    echo "[local] normalize-pairs shard $i"
    eval "$(stage_command normalize-pairs "$i")"
  done
  for i in $(seq 0 $((MOLEDIT_SHARDS - 1))); do
    echo "[local] molecule-cache shard $i"
    eval "$(stage_command molecule-cache "$i")"
  done
  for i in $(seq 0 $((MOLEDIT_SHARDS - 1))); do
    echo "[local] pair-features shard $i"
    eval "$(stage_command pair-features "$i")"
  done
  echo "[local] finalize"
  eval "$(stage_command finalize)"
}

print_dry_run() {
  print_config
  echo
  echo "Normalize array command:"
  stage_command normalize-pairs '${SLURM_ARRAY_TASK_ID}'
  echo
  echo
  echo "Molecule-cache array command:"
  stage_command molecule-cache '${SLURM_ARRAY_TASK_ID}'
  echo
  echo
  echo "Pair-features array command:"
  stage_command pair-features '${SLURM_ARRAY_TASK_ID}'
  echo
  echo
  echo "Finalize command:"
  stage_command finalize
  echo
}

if [[ "$MOLEDIT_DRY_RUN" == "1" ]]; then
  print_dry_run
  exit 0
fi

mkdir -p "$MOLEDIT_OUTPUT_DIR/logs"

if [[ "$MOLEDIT_LOCAL" == "1" ]]; then
  run_local
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Re-run with MOLEDIT_LOCAL=1 for serial local execution or --dry-run to inspect commands." >&2
  exit 1
fi

if [[ ! -s "$MOLEDIT_RAW_INPUT" ]]; then
  echo "Raw input not found or empty: $MOLEDIT_RAW_INPUT" >&2
  exit 1
fi

SBATCH_COMMON=(
  --parsable
  --account="$MOLEDIT_SLURM_ACCOUNT"
  --job-name="$MOLEDIT_JOB_NAME"
  --cpus-per-task="$MOLEDIT_CPUS_PER_TASK"
  --mem="$MOLEDIT_MEM"
  --time="$MOLEDIT_TIME"
)

if [[ -n "$MOLEDIT_SLURM_PARTITION" ]]; then
  SBATCH_COMMON+=(--partition="$MOLEDIT_SLURM_PARTITION")
fi

array_spec="0-$((MOLEDIT_SHARDS - 1))"
normalize_job="$(
  sbatch "${SBATCH_COMMON[@]}" \
    --array="$array_spec" \
    --output="$MOLEDIT_OUTPUT_DIR/logs/normalize_%A_%a.out" \
    --error="$MOLEDIT_OUTPUT_DIR/logs/normalize_%A_%a.err" \
    --wrap "$(stage_command normalize-pairs '${SLURM_ARRAY_TASK_ID}')"
)"

cache_job="$(
  sbatch "${SBATCH_COMMON[@]}" \
    --array="$array_spec" \
    --dependency="afterok:$normalize_job" \
    --output="$MOLEDIT_OUTPUT_DIR/logs/cache_%A_%a.out" \
    --error="$MOLEDIT_OUTPUT_DIR/logs/cache_%A_%a.err" \
    --wrap "$(stage_command molecule-cache '${SLURM_ARRAY_TASK_ID}')"
)"

pair_job="$(
  sbatch "${SBATCH_COMMON[@]}" \
    --array="$array_spec" \
    --dependency="afterok:$cache_job" \
    --output="$MOLEDIT_OUTPUT_DIR/logs/pair_%A_%a.out" \
    --error="$MOLEDIT_OUTPUT_DIR/logs/pair_%A_%a.err" \
    --wrap "$(stage_command pair-features '${SLURM_ARRAY_TASK_ID}')"
)"

finalize_job="$(
  sbatch "${SBATCH_COMMON[@]}" \
    --dependency="afterok:$pair_job" \
    --output="$MOLEDIT_OUTPUT_DIR/logs/finalize_%j.out" \
    --error="$MOLEDIT_OUTPUT_DIR/logs/finalize_%j.err" \
    --wrap "$(stage_command finalize)"
)"

print_config
echo "Submitted normalize job: $normalize_job"
echo "Submitted molecule-cache job: $cache_job"
echo "Submitted pair-features job: $pair_job"
echo "Submitted finalize job: $finalize_job"
