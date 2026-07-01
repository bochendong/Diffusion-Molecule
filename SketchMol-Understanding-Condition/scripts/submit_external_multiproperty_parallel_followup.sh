#!/usr/bin/env bash
# Submit work that can run while waiting for external benchmark results.
#
# Dependency graph:
#   sourcefix oracle -> C-MuMO official re-eval
#   MuMO IND-hard GraphEdit sweeps -> MuMO IND-hard oracle -> re-evals

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_SOURCEFIX_ORACLE="${SUCC_FOLLOWUP_RUN_SOURCEFIX_ORACLE:-1}"
RUN_CMUMO_RERUN="${SUCC_FOLLOWUP_RUN_CMUMO_RERUN:-1}"
RUN_IND_HARD_SWEEP="${SUCC_FOLLOWUP_RUN_IND_HARD_SWEEP:-1}"

OLD_ORACLE="${SUCC_FOLLOWUP_OLD_ORACLE:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
NEW_ORACLE="${SUCC_FOLLOWUP_NEW_ORACLE:-SketchMol-Understanding-Condition/outputs/external_oracle_build_sourcefix_v1/generated_properties.csv}"
SOURCEFIX_WORK_DIR="${SUCC_FOLLOWUP_SOURCEFIX_WORK_DIR:-$(dirname "$NEW_ORACLE")}"
SOURCEFIX_INPUT_CSV="${SUCC_FOLLOWUP_SOURCEFIX_INPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_mumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/external_multiproperty_rows.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv}"

echo "External benchmark parallel follow-up"
echo "  run_sourcefix_oracle=$RUN_SOURCEFIX_ORACLE"
echo "  run_cmumo_rerun=$RUN_CMUMO_RERUN"
echo "  run_ind_hard_sweep=$RUN_IND_HARD_SWEEP"
echo "  old_oracle=$OLD_ORACLE"
echo "  new_oracle=$NEW_ORACLE"

sourcefix_job_id="${SUCC_FOLLOWUP_SOURCEFIX_JOB_ID:-}"
if [[ "$RUN_SOURCEFIX_ORACLE" == "1" ]]; then
  echo
  echo "=== submitting sourcefix oracle ==="
  sourcefix_output="$(
    SUCC_ORACLE_SLURM_JOB_NAME=succ-ext-oracle-sourcefix \
    SUCC_ORACLE_WORK_DIR="$SOURCEFIX_WORK_DIR" \
    SUCC_ORACLE_OUTPUT_CSV="$NEW_ORACLE" \
    SUCC_ORACLE_MERGE_PROPERTIES_CSV="$OLD_ORACLE" \
    SUCC_ORACLE_INPUT_CSV="$SOURCEFIX_INPUT_CSV" \
    bash "$PROJECT_DIR/scripts/submit_external_multiproperty_generated_oracle_pipeline.sh" 2>&1
  )"
  echo "$sourcefix_output"
  sourcefix_job_id="$(echo "$sourcefix_output" | sed -n 's/.*external_oracle_build_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$sourcefix_job_id" ]]; then
    echo "WARN: could not parse sourcefix oracle job id; C-MuMO rerun will not be dependency-gated." >&2
  fi
fi

if [[ "$RUN_CMUMO_RERUN" == "1" ]]; then
  echo
  echo "=== submitting C-MuMO official rerun ==="
  cmumo_dependency=""
  if [[ -n "$sourcefix_job_id" ]]; then
    cmumo_dependency="afterok:$sourcefix_job_id"
  elif [[ -n "${SUCC_FOLLOWUP_CMUMO_DEPENDENCY:-}" ]]; then
    cmumo_dependency="$SUCC_FOLLOWUP_CMUMO_DEPENDENCY"
  fi
  SUCC_OFFICIAL_RUN_MUMO=0 \
  SUCC_OFFICIAL_RUN_CMUMO=1 \
  SUCC_OFFICIAL_SLURM_DEPENDENCY="$cmumo_dependency" \
  SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV="$NEW_ORACLE" \
  SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV="$NEW_ORACLE" \
  bash "$PROJECT_DIR/scripts/submit_external_multiproperty_official_suite.sh" || true
fi

if [[ "$RUN_IND_HARD_SWEEP" == "1" ]]; then
  echo
  echo "=== submitting MuMO IND-hard sweep ==="
  bash "$PROJECT_DIR/scripts/submit_external_mumo_ind_hard_sweep.sh" || true
fi

echo
echo "External benchmark parallel follow-up submitted."
echo "  sourcefix_oracle_job=${sourcefix_job_id:-none}"
