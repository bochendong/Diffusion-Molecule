#!/usr/bin/env bash
# Submit MuMO/C-MuMO official-style source-conditioned benchmark jobs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MUMO_SOURCE_FILE="${SUCC_OFFICIAL_MUMO_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
CMUMO_SOURCE_FILE="${SUCC_OFFICIAL_CMUMO_SOURCE_FILE:-/scratch/bdong/datasets/Diffusion-Molecule/external/cmumo/test.json}"
MAX_ROWS_PER_TASK="${SUCC_OFFICIAL_MAX_ROWS_PER_TASK:-200}"
TASK_SPLIT="${SUCC_OFFICIAL_TASK_SPLIT:-all}"
RUN_MUMO="${SUCC_OFFICIAL_RUN_MUMO:-1}"
RUN_CMUMO="${SUCC_OFFICIAL_RUN_CMUMO:-1}"
RUN_COPY_SANITY="${SUCC_OFFICIAL_RUN_COPY_SANITY:-1}"
RUN_GRAPH_EDIT="${SUCC_OFFICIAL_RUN_GRAPH_EDIT:-1}"
BUILD_ORACLE_CSV="${SUCC_OFFICIAL_BUILD_ORACLE_CSV:-1}"
DEPENDENCY="${SUCC_OFFICIAL_SLURM_DEPENDENCY:-${SUCC_SLURM_DEPENDENCY:-}}"
FORCE_EXPORT="${SUCC_OFFICIAL_FORCE_EXPORT:-0}"
MUMO_COPY_OUTPUT_DIR="${SUCC_OFFICIAL_MUMO_COPY_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_official_copy_sanity_v1}"
MUMO_GRAPH_OUTPUT_DIR="${SUCC_OFFICIAL_MUMO_GRAPH_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_official_graph_edit_heuristic_2step_v1}"
CMUMO_COPY_OUTPUT_DIR="${SUCC_OFFICIAL_CMUMO_COPY_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_cmumo_official_copy_sanity_v1}"
CMUMO_GRAPH_OUTPUT_DIR="${SUCC_OFFICIAL_CMUMO_GRAPH_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1}"

run_step() {
  local label="$1"
  shift
  echo
  echo "=== submitting $label ==="
  if ! "$@"; then
    echo "WARN: $label failed to submit; continuing official suite." >&2
  fi
}

submit_copy_sanity() {
  local suite="$1"
  local source_file="$2"
  local output_dir="$3"
  local job_name="$4"
  (
    export SUCC_EXTERNAL_SOURCE_COPY_SOURCE_FILE="$source_file"
    export SUCC_EXTERNAL_SOURCE_COPY_SUITE="$suite"
    export SUCC_EXTERNAL_SOURCE_COPY_TASK_SPLIT="$TASK_SPLIT"
    export SUCC_EXTERNAL_SOURCE_COPY_MAX_ROWS_PER_TASK="$MAX_ROWS_PER_TASK"
    export SUCC_EXTERNAL_SOURCE_COPY_OUTPUT_DIR="$output_dir"
    export SUCC_EXTERNAL_SOURCE_COPY_SLURM_JOB_NAME="$job_name"
    export SUCC_EXTERNAL_SOURCE_COPY_BUILD_ORACLE_CSV="$BUILD_ORACLE_CSV"
    export SUCC_EXTERNAL_SOURCE_COPY_SLURM_DEPENDENCY="$DEPENDENCY"
    export SUCC_EXTERNAL_SOURCE_COPY_FORCE_EXPORT="$FORCE_EXPORT"
    bash "$PROJECT_DIR/scripts/submit_external_mumo_source_copy_sanity.sh"
  )
}

submit_graph_edit() {
  local suite="$1"
  local source_file="$2"
  local output_dir="$3"
  local job_name="$4"
  local require_direct="$5"
  (
    export SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE="$source_file"
    export SUCC_EXTERNAL_GRAPH_EDIT_SUITE="$suite"
    export SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT="$TASK_SPLIT"
    export SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK="$MAX_ROWS_PER_TASK"
    export SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR="$output_dir"
    export SUCC_EXTERNAL_GRAPH_EDIT_SLURM_JOB_NAME="$job_name"
    export SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS="$require_direct"
    export SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES="${SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES:-20}"
    export SUCC_EXTERNAL_GRAPH_EDIT_BUILD_ORACLE_CSV="$BUILD_ORACLE_CSV"
    export SUCC_EXTERNAL_GRAPH_EDIT_SLURM_DEPENDENCY="$DEPENDENCY"
    export SUCC_EXTERNAL_GRAPH_EDIT_FORCE_EXPORT="$FORCE_EXPORT"
    bash "$PROJECT_DIR/scripts/submit_direct_smiles_external_mumo_graph_edit_heuristic_2step.sh"
  )
}

echo "External multi-property official suite"
echo "  mumo_source_file=$MUMO_SOURCE_FILE"
echo "  cmumo_source_file=$CMUMO_SOURCE_FILE"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  task_split=$TASK_SPLIT"
echo "  build_oracle_csv=$BUILD_ORACLE_CSV"
echo "  force_export=$FORCE_EXPORT"
echo "  generated_properties_csv=${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-none}"
echo "  source_properties_csv=${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-none}"
echo "  dependency=${DEPENDENCY:-none}"

if [[ "$RUN_MUMO" == "1" ]]; then
  if [[ "$RUN_COPY_SANITY" == "1" ]]; then
    run_step "MuMO source/target-copy sanity" submit_copy_sanity \
      mumo \
      "$MUMO_SOURCE_FILE" \
      "$MUMO_COPY_OUTPUT_DIR" \
      succ-mumo-official-copy
  fi
  if [[ "$RUN_GRAPH_EDIT" == "1" ]]; then
    run_step "MuMO GraphEditDSL top20" submit_graph_edit \
      mumo \
      "$MUMO_SOURCE_FILE" \
      "$MUMO_GRAPH_OUTPUT_DIR" \
      succ-mumo-official-graph \
      1
  fi
fi

if [[ "$RUN_CMUMO" == "1" ]]; then
  if [[ "$RUN_COPY_SANITY" == "1" ]]; then
    run_step "C-MuMO source/target-copy sanity" submit_copy_sanity \
      cmumo \
      "$CMUMO_SOURCE_FILE" \
      "$CMUMO_COPY_OUTPUT_DIR" \
      succ-cmumo-official-copy
  fi
  if [[ "$RUN_GRAPH_EDIT" == "1" ]]; then
    run_step "C-MuMO GraphEditDSL top20 source-only" submit_graph_edit \
      cmumo \
      "$CMUMO_SOURCE_FILE" \
      "$CMUMO_GRAPH_OUTPUT_DIR" \
      succ-cmumo-official-graph \
      0
  fi
fi

echo
echo "Official suite submission finished."
