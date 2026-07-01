#!/usr/bin/env bash
# Submit external multi-property oracle re-evaluation on Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

PREDICTION_CSV="${SUCC_EXTERNAL_REEVAL_PREDICTION_CSV:-}"
OUTPUT_DIR="${SUCC_EXTERNAL_REEVAL_OUTPUT_DIR:-}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_REEVAL_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"
JOB_NAME="${SUCC_EXTERNAL_REEVAL_SLURM_JOB_NAME:-succ-ext-reeval}"
ACCOUNT="${SUCC_EXTERNAL_REEVAL_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_EXTERNAL_REEVAL_SLURM_TIME:-01:00:00}"
CPUS="${SUCC_EXTERNAL_REEVAL_SLURM_CPUS:-2}"
MEM="${SUCC_EXTERNAL_REEVAL_SLURM_MEM:-16G}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_EXTERNAL_REEVAL_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
DEPENDENCY="${SUCC_EXTERNAL_REEVAL_SLURM_DEPENDENCY:-${SUCC_SLURM_DEPENDENCY:-}}"

if [[ -z "$PREDICTION_CSV" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_REEVAL_PREDICTION_CSV." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_REEVAL_OUTPUT_DIR." >&2
  exit 2
fi
if [[ -z "$GENERATED_PROPERTIES_CSV" ]]; then
  echo "ERROR: set SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV." >&2
  exit 2
fi
if [[ -z "$DEPENDENCY" && ! -f "$PREDICTION_CSV" ]]; then
  echo "ERROR: prediction CSV does not exist and no dependency was provided: $PREDICTION_CSV" >&2
  exit 2
fi
if [[ -z "$DEPENDENCY" && ! -f "$GENERATED_PROPERTIES_CSV" ]]; then
  echo "ERROR: oracle CSV does not exist and no dependency was provided: $GENERATED_PROPERTIES_CSV" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

echo "Submitting external multi-property oracle re-eval"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  generated_properties_csv=$GENERATED_PROPERTIES_CSV"
echo "  source_properties_csv=${SOURCE_PROPERTIES_CSV:-none}"
echo "  dependency=${DEPENDENCY:-none}"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL,SUCC_EXTERNAL_REEVAL_PREDICTION_CSV="$PREDICTION_CSV",SUCC_EXTERNAL_REEVAL_OUTPUT_DIR="$OUTPUT_DIR",SUCC_EXTERNAL_REEVAL_GENERATED_PROPERTIES_CSV="$GENERATED_PROPERTIES_CSV",SUCC_EXTERNAL_REEVAL_SOURCE_PROPERTIES_CSV="$SOURCE_PROPERTIES_CSV",SUCC_PYTHON_BIN="$PYTHON_BIN"
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_multiproperty_re_eval.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit external multi-property re-eval." >&2
  exit 1
fi

echo "external_multiproperty_reeval_job=$job_id"
