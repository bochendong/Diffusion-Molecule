#!/usr/bin/env bash
# Submit C-MuMO dedicated oracle build, then rerun the official C-MuMO suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

RUN_ORACLE="${SUCC_CMUMO_FIX_RUN_ORACLE:-1}"
RUN_RERUN="${SUCC_CMUMO_FIX_RUN_RERUN:-1}"
WORK_DIR="${SUCC_CMUMO_ORACLE_WORK_DIR:-SketchMol-Understanding-Condition/outputs/external_oracle_build_cmumo_v1}"
OUTPUT_CSV="${SUCC_CMUMO_ORACLE_OUTPUT_CSV:-$WORK_DIR/generated_properties.csv}"
COPY_OUTPUT_DIR="${SUCC_CMUMO_FIX_COPY_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_cmumo_official_copy_sanity_cmumo_oracle_v1}"
GRAPH_OUTPUT_DIR="${SUCC_CMUMO_FIX_GRAPH_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_cmumo_oracle_v1}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
ACCOUNT="${SUCC_CMUMO_ORACLE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_CMUMO_ORACLE_SLURM_TIME:-06:00:00}"
CPUS="${SUCC_CMUMO_ORACLE_SLURM_CPUS:-4}"
MEM="${SUCC_CMUMO_ORACLE_SLURM_MEM:-32G}"
PARTITION="${SUCC_CMUMO_ORACLE_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
DEPENDENCY="${SUCC_CMUMO_ORACLE_SLURM_DEPENDENCY:-${SUCC_SLURM_DEPENDENCY:-}}"

mkdir -p "$WORK_DIR" "$LOG_DIR"

echo "Submitting C-MuMO dedicated oracle fix"
echo "  run_oracle=$RUN_ORACLE"
echo "  run_rerun=$RUN_RERUN"
echo "  work_dir=$WORK_DIR"
echo "  output_csv=$OUTPUT_CSV"
echo "  copy_output_dir=$COPY_OUTPUT_DIR"
echo "  graph_output_dir=$GRAPH_OUTPUT_DIR"
echo "  dependency=${DEPENDENCY:-none}"

oracle_job_id="${SUCC_CMUMO_FIX_ORACLE_JOB_ID:-}"
if [[ "$RUN_ORACLE" == "1" ]]; then
  sbatch_args=(
    --account="$ACCOUNT"
    --job-name="${SUCC_CMUMO_ORACLE_SLURM_JOB_NAME:-succ-cmumo-oracle-v1}"
    --time="$TIME"
    --mem="$MEM"
    --cpus-per-task="$CPUS"
    --output="$LOG_DIR/%x-%j.log"
    --export=ALL,SUCC_CMUMO_ORACLE_WORK_DIR="$WORK_DIR",SUCC_CMUMO_ORACLE_OUTPUT_CSV="$OUTPUT_CSV",SUCC_PYTHON_BIN="$PYTHON_BIN",SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN"
  )
  if [[ -n "$PARTITION" ]]; then
    sbatch_args+=(--partition="$PARTITION")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
  fi
  output="$(sbatch "${sbatch_args[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_external_cmumo_dedicated_oracle_fix.sh'")"
  echo "$output"
  oracle_job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$oracle_job_id" ]]; then
    echo "ERROR: failed to parse oracle job id." >&2
    exit 1
  fi
fi

if [[ "$RUN_RERUN" == "1" ]]; then
  rerun_dependency="${SUCC_CMUMO_FIX_RERUN_DEPENDENCY:-}"
  if [[ -n "$oracle_job_id" ]]; then
    rerun_dependency="afterok:$oracle_job_id"
  elif [[ -z "$rerun_dependency" ]]; then
    echo "WARN: rerun has no oracle dependency; make sure $OUTPUT_CSV already exists." >&2
  fi
  SUCC_OFFICIAL_RUN_MUMO=0 \
  SUCC_OFFICIAL_RUN_CMUMO=1 \
  SUCC_OFFICIAL_BUILD_ORACLE_CSV=0 \
  SUCC_OFFICIAL_FORCE_EXPORT=1 \
  SUCC_OFFICIAL_CMUMO_COPY_OUTPUT_DIR="$COPY_OUTPUT_DIR" \
  SUCC_OFFICIAL_CMUMO_GRAPH_OUTPUT_DIR="$GRAPH_OUTPUT_DIR" \
  SUCC_OFFICIAL_SLURM_DEPENDENCY="$rerun_dependency" \
  SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV="$OUTPUT_CSV" \
  SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV="$OUTPUT_CSV" \
  bash "$PROJECT_DIR/scripts/submit_external_multiproperty_official_suite.sh" || true
fi

echo
echo "C-MuMO dedicated oracle fix submitted."
echo "  oracle_job=${oracle_job_id:-none}"
echo "  generated_properties_csv=$OUTPUT_CSV"
echo "  copy_output_dir=$COPY_OUTPUT_DIR"
echo "  graph_output_dir=$GRAPH_OUTPUT_DIR"
