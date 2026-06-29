#!/usr/bin/env bash
# Submit a broad MuMO sweep for long unattended runs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

run_step() {
  local name="$1"
  local script="$2"
  if [[ "${!name:-1}" == "0" ]]; then
    echo "Skipping $script because $name=0"
    return 0
  fi
  echo
  echo "=== submitting $script ==="
  if ! bash "$PROJECT_DIR/scripts/$script"; then
    echo "WARN: $script failed to submit; continuing flight sweep." >&2
  fi
}

echo "MuMO flight sweep"
echo "  set any SUCC_FLIGHT_* env var to 0 to skip a branch"

run_step SUCC_FLIGHT_GRAPH_SIM submit_direct_smiles_external_mumo_graph_edit_similarity_anchor.sh
run_step SUCC_FLIGHT_GRAPH_HEUR2 submit_direct_smiles_external_mumo_graph_edit_heuristic_2step.sh
run_step SUCC_FLIGHT_RICH_X2 submit_direct_smiles_external_mumo_agentic_revise_rich_x2.sh
run_step SUCC_FLIGHT_SOURCE_SFT_LONG submit_direct_smiles_external_mumo_source_edit_sft_long.sh
run_step SUCC_FLIGHT_SOURCE_RL_OFFICIAL submit_direct_smiles_external_mumo_source_edit_rl_official_rerun.sh
run_step SUCC_FLIGHT_SOURCE_RL_HIGHSIM submit_direct_smiles_external_mumo_source_edit_rl_highsim.sh

echo
echo "Flight sweep submission finished."
