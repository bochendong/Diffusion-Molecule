#!/usr/bin/env bash
# Submit ADMET-AI generated-property oracle build on Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

INPUT_CSV="${SUCC_ORACLE_INPUT_CSV:-}"
OUTPUT_CSV="${SUCC_ORACLE_OUTPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
WORK_DIR="${SUCC_ORACLE_WORK_DIR:-$(dirname "$OUTPUT_CSV")}"
JOB_NAME="${SUCC_ORACLE_SLURM_JOB_NAME:-succ-ext-oracle}"
ACCOUNT="${SUCC_ORACLE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_ORACLE_SLURM_TIME:-04:00:00}"
CPUS="${SUCC_ORACLE_SLURM_CPUS:-4}"
MEM="${SUCC_ORACLE_SLURM_MEM:-32G}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_ORACLE_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"

if [[ -z "$INPUT_CSV" ]]; then
  echo "ERROR: set SUCC_ORACLE_INPUT_CSV to prediction CSV path(s)." >&2
  exit 2
fi

mkdir -p "$WORK_DIR" "$LOG_DIR"

echo "Submitting external multiproperty oracle build"
echo "  input_csv=$INPUT_CSV"
echo "  output_csv=$OUTPUT_CSV"
echo "  work_dir=$WORK_DIR"
echo "  python=$PYTHON_BIN"
echo "  admet_python=$ADMET_PYTHON_BIN"

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL,SUCC_ORACLE_INPUT_CSV="$INPUT_CSV",SUCC_ORACLE_OUTPUT_CSV="$OUTPUT_CSV",SUCC_ORACLE_WORK_DIR="$WORK_DIR",SUCC_PYTHON_BIN="$PYTHON_BIN",SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN"
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh'")"
echo "$output"
job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit oracle build job." >&2
  exit 1
fi

echo "external_oracle_build_job=$job_id"
