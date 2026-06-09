#!/usr/bin/env bash
# Shared defaults for the source-neighbor multi-property edit dataset (v1).

SMMED_DEFAULT_OUTPUT_DIR="${SMMED_DEFAULT_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1}"
SMMED_DEFAULT_CONDITION_ROWS="${SMMED_DEFAULT_OUTPUT_DIR}/condition_rows.csv"
SMMED_DEFAULT_EDIT_MANIFEST="${SMMED_DEFAULT_OUTPUT_DIR}/diffusion_edit_manifest.csv"
SMMED_DEFAULT_BASELINE_CSV="${SMMED_DEFAULT_OUTPUT_DIR}/baseline_variants.csv"
SMMED_DEFAULT_MOLECULE_DB="${SMMED_DEFAULT_OUTPUT_DIR}/molecule_database.csv"

SUCC_DEFAULT_FEATURES_DIR="${SUCC_DEFAULT_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_source_neighbor_v1}"
SUCC_DEFAULT_BENCHMARK_DIR="${SUCC_DEFAULT_BENCHMARK_DIR:-$SMMED_DEFAULT_OUTPUT_DIR/benchmark_hf_vlm_source_neighbor_v1}"
SUCC_DEFAULT_UNIFIED_OUTPUT_DIR="${SUCC_DEFAULT_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_generation_3m_source_neighbor_v1}"
SUCC_DEFAULT_UNIVIDEO_OUTPUT_DIR="${SUCC_DEFAULT_UNIVIDEO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_source_neighbor_v2_residual_ink}"
SUCC_CANONICAL_V2_OUTPUT_DIR="${SUCC_CANONICAL_V2_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink}"
SUCC_CANONICAL_V2_IMAGE_VAE_CHECKPOINT="${SUCC_CANONICAL_V2_IMAGE_VAE_CHECKPOINT:-$SUCC_CANONICAL_V2_OUTPUT_DIR/molecule_image_vae/molecule_image_vae.pt}"

export_smmed_source_neighbor_defaults() {
  export SMMED_OUTPUT_DIR="${SMMED_OUTPUT_DIR:-$SMMED_DEFAULT_OUTPUT_DIR}"
  export SMMED_PAIRING_STRATEGY="${SMMED_PAIRING_STRATEGY:-source_neighbor}"
  export SMMED_SOURCE_NEIGHBOR_MIN_TANIMOTO="${SMMED_SOURCE_NEIGHBOR_MIN_TANIMOTO:-0.4}"
  export SMMED_SOURCE_NEIGHBOR_MAX_TANIMOTO="${SMMED_SOURCE_NEIGHBOR_MAX_TANIMOTO:-0.95}"
  export SMMED_MIN_SOURCE_NEIGHBORS_T04="${SMMED_MIN_SOURCE_NEIGHBORS_T04:-2}"
  export SMMED_MIN_SIMILARITY="${SMMED_MIN_SIMILARITY:-0.4}"
  export SMMED_MAX_SIMILARITY="${SMMED_MAX_SIMILARITY:-0.95}"
  export SMMED_CONDITION_CANDIDATE_DIAGNOSTICS="${SMMED_CONDITION_CANDIDATE_DIAGNOSTICS:-1}"
  export SMMED_MIN_STRICT_CANDIDATES_T04="${SMMED_MIN_STRICT_CANDIDATES_T04:-1}"
  export SMMED_ORACLE_FILTER_SPLITS="${SMMED_ORACLE_FILTER_SPLITS:-eval}"
  export SMMED_DIFFUSION_MIN_SOURCE_TANIMOTO="${SMMED_DIFFUSION_MIN_SOURCE_TANIMOTO:-0.4}"
  export SMMED_DIFFUSION_MAX_SOURCE_TANIMOTO="${SMMED_DIFFUSION_MAX_SOURCE_TANIMOTO:-0.95}"
}

export_succ_edit_quality_defaults() {
  export SUCC_MIN_EDIT_SOURCE_TANIMOTO="${SUCC_MIN_EDIT_SOURCE_TANIMOTO:-0.4}"
  export SUCC_REQUIRE_EDIT_QUALITY_COLUMNS="${SUCC_REQUIRE_EDIT_QUALITY_COLUMNS:-1}"
  export SUCC_REQUIRE_EVAL_ORACLE_STRICT="${SUCC_REQUIRE_EVAL_ORACLE_STRICT:-1}"
}

run_source_neighbor_dataset_build() {
  local dataset_project_dir="$1"
  export_smmed_source_neighbor_defaults
  bash "$dataset_project_dir/scripts/run_build_source_neighbor_dataset.sh"
}
