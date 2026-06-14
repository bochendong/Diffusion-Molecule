#!/usr/bin/env bash
# Submit a fixed v1/v3/v4 materializer sweep for zero-source SUCC benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

METHODS="${SUCC_SWEEP_MATERIALIZED_METHODS:-latent_nearest,latent_property_rerank,property_nearest}"
PROPERTY_RERANK_CANDIDATES="${SUCC_SWEEP_PROPERTY_RERANK_CANDIDATES:-4096}"
PROPERTY_RERANK_WEIGHT="${SUCC_SWEEP_PROPERTY_RERANK_WEIGHT:-10}"
STRICT_RERANK_WEIGHT="${SUCC_SWEEP_STRICT_RERANK_WEIGHT:-100}"
LATENT_RERANK_WEIGHT="${SUCC_SWEEP_LATENT_RERANK_WEIGHT:-1}"

echo "Submitting SUCC zero-source materializer sweep"
echo "  methods=$METHODS"
echo "  property_rerank_candidates=$PROPERTY_RERANK_CANDIDATES"
echo "  property_rerank_weight=$PROPERTY_RERANK_WEIGHT"
echo "  strict_rerank_weight=$STRICT_RERANK_WEIGHT"
echo "  latent_rerank_weight=$LATENT_RERANK_WEIGHT"

submit_pair() {
  local label="$1"
  local model_output_dir="$2"
  local checkpoint="$model_output_dir/univideo_molecule/univideo_molecule_generation.pt"

  echo
  echo "Materializer sweep: $label"
  echo "  model_output_dir=$model_output_dir"
  echo "  checkpoint=$checkpoint"

  SUCC_DENOVO_MODEL_OUTPUT_DIR="$model_output_dir" \
  SUCC_DENOVO_RESUME_CHECKPOINT="$checkpoint" \
  SUCC_DENOVO_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_2p7p_materializer_sweep_${label}" \
  SUCC_DENOVO_BENCHMARK_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_2p7p_materializer_sweep_${label}/benchmark_ours" \
  SUCC_DENOVO_MATERIALIZED_METHODS="$METHODS" \
  SUCC_DENOVO_PROPERTY_RERANK_CANDIDATES="$PROPERTY_RERANK_CANDIDATES" \
  SUCC_DENOVO_PROPERTY_RERANK_WEIGHT="$PROPERTY_RERANK_WEIGHT" \
  SUCC_DENOVO_STRICT_RERANK_WEIGHT="$STRICT_RERANK_WEIGHT" \
  SUCC_DENOVO_LATENT_RERANK_WEIGHT="$LATENT_RERANK_WEIGHT" \
  SUCC_DENOVO_SLURM_JOB_NAME="succ-2p7p-mat-${label}" \
  bash "$SCRIPT_DIR/submit_denovo_2p7p_ours_benchmark.sh"

  SUCC_OOD_MODEL_OUTPUT_DIR="$model_output_dir" \
  SUCC_OOD_RESUME_CHECKPOINT="$checkpoint" \
  SUCC_OOD_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_ood_materializer_sweep_${label}" \
  SUCC_OOD_BENCHMARK_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_ood_materializer_sweep_${label}/benchmark_ours" \
  SUCC_OOD_MATERIALIZED_METHODS="$METHODS" \
  SUCC_OOD_PROPERTY_RERANK_CANDIDATES="$PROPERTY_RERANK_CANDIDATES" \
  SUCC_OOD_PROPERTY_RERANK_WEIGHT="$PROPERTY_RERANK_WEIGHT" \
  SUCC_OOD_STRICT_RERANK_WEIGHT="$STRICT_RERANK_WEIGHT" \
  SUCC_OOD_LATENT_RERANK_WEIGHT="$LATENT_RERANK_WEIGHT" \
  SUCC_OOD_SLURM_JOB_NAME="succ-ood-mat-${label}" \
  bash "$SCRIPT_DIR/submit_denovo_ood_ours_benchmark.sh"
}

submit_pair \
  "v1" \
  "SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1"

submit_pair \
  "v3" \
  "SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded"

submit_pair \
  "v4" \
  "SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1"

echo
echo "Materializer sweep submitted."
