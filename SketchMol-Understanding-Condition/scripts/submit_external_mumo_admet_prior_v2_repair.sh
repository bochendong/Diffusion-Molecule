#!/usr/bin/env bash
# MuMO ADMET-prior v2: admet_hybrid ranking + IND-hard balanced tuning.
#
# v1 used similarity_first, which ranks by local proxy (qed/plogp/sa) and largely
# ignores BBBP/DRD2/HIA/mutagenicity. v2 keeps ADMET-prior generation but ranks
# Sim>=0.4 candidates by admet_prior_score before local proxy tie-breakers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUCC_MUMO_ADMET_OUTPUT_DIR="${SUCC_MUMO_ADMET_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/external_mumo_graph_edit_admet_prior_v2}"
export SUCC_MUMO_ADMET_ORACLE_OUTPUT_CSV="${SUCC_MUMO_ADMET_ORACLE_OUTPUT_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_mumo_admet_prior_v2/generated_properties.csv}"
export SUCC_MUMO_ADMET_ORACLE_WORK_DIR="${SUCC_MUMO_ADMET_ORACLE_WORK_DIR:-$(dirname "$SUCC_MUMO_ADMET_ORACLE_OUTPUT_CSV")}"
export SUCC_MUMO_ADMET_GRAPH_JOB_NAME="${SUCC_MUMO_ADMET_GRAPH_JOB_NAME:-succ-mumo-admet-v2-graph}"
export SUCC_MUMO_ADMET_ORACLE_JOB_NAME="${SUCC_MUMO_ADMET_ORACLE_JOB_NAME:-succ-mumo-admet-v2-oracle}"
export SUCC_MUMO_ADMET_REEVAL_JOB_NAME="${SUCC_MUMO_ADMET_REEVAL_JOB_NAME:-succ-mumo-admet-v2-reeval}"

export SUCC_MUMO_ADMET_SELECTION_MODE="${SUCC_MUMO_ADMET_SELECTION_MODE:-admet_hybrid}"
export SUCC_MUMO_ADMET_LOCAL_SUCCESS_FRACTION="${SUCC_MUMO_ADMET_LOCAL_SUCCESS_FRACTION:-0.85}"
export SUCC_MUMO_ADMET_PRIOR_WEIGHT="${SUCC_MUMO_ADMET_PRIOR_WEIGHT:-110}"
export SUCC_MUMO_ADMET_PROPERTY_WEIGHT="${SUCC_MUMO_ADMET_PROPERTY_WEIGHT:-90}"
export SUCC_MUMO_ADMET_DISTANCE_WEIGHT="${SUCC_MUMO_ADMET_DISTANCE_WEIGHT:-8}"
export SUCC_MUMO_ADMET_SIMILARITY_WEIGHT="${SUCC_MUMO_ADMET_SIMILARITY_WEIGHT:-100}"
export SUCC_MUMO_ADMET_SIMILARITY_BONUS="${SUCC_MUMO_ADMET_SIMILARITY_BONUS:-160}"
export SUCC_MUMO_ADMET_BEAM_SIZE="${SUCC_MUMO_ADMET_BEAM_SIZE:-192}"
export SUCC_MUMO_ADMET_MAX_CANDIDATES_PER_ROW="${SUCC_MUMO_ADMET_MAX_CANDIDATES_PER_ROW:-8192}"
export SUCC_MUMO_ADMET_TOP_K_CANDIDATES="${SUCC_MUMO_ADMET_TOP_K_CANDIDATES:-40}"

exec bash "$SCRIPT_DIR/submit_external_mumo_admet_prior_repair.sh"
