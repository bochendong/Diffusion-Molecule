#!/usr/bin/env bash
# Submit random-shortlist sanity checks for the zero-source hybrid materializer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

LABEL="${SUCC_SANITY_LABEL:-v1}"
MODEL_OUTPUT_DIR="${SUCC_SANITY_MODEL_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1}"
CHECKPOINT="${SUCC_SANITY_RESUME_CHECKPOINT:-$MODEL_OUTPUT_DIR/univideo_molecule/univideo_molecule_generation.pt}"
METHODS="${SUCC_SANITY_MATERIALIZED_METHODS:-latent_property_rerank,random_property_rerank,property_nearest}"
CANDIDATE_SIZES_STR="${SUCC_SANITY_PROPERTY_RERANK_CANDIDATES_LIST:-64 128 256 512 1024 4096}"
PROPERTY_RERANK_WEIGHT="${SUCC_SANITY_PROPERTY_RERANK_WEIGHT:-10}"
STRICT_RERANK_WEIGHT="${SUCC_SANITY_STRICT_RERANK_WEIGHT:-100}"
LATENT_RERANK_WEIGHT="${SUCC_SANITY_LATENT_RERANK_WEIGHT:-1}"
RANDOM_RERANK_SEED="${SUCC_SANITY_RANDOM_RERANK_SEED:-13}"

echo "Submitting SUCC zero-source materializer sanity sweep"
echo "  label=$LABEL"
echo "  model_output_dir=$MODEL_OUTPUT_DIR"
echo "  checkpoint=$CHECKPOINT"
echo "  methods=$METHODS"
echo "  candidate_sizes=$CANDIDATE_SIZES_STR"
echo "  property_rerank_weight=$PROPERTY_RERANK_WEIGHT"
echo "  strict_rerank_weight=$STRICT_RERANK_WEIGHT"
echo "  latent_rerank_weight=$LATENT_RERANK_WEIGHT"
echo "  random_rerank_seed=$RANDOM_RERANK_SEED"

for candidate_size in $CANDIDATE_SIZES_STR; do
  candidate_size="$(echo "$candidate_size" | tr -d ',')"
  if [[ -z "$candidate_size" ]]; then
    continue
  fi
  suffix="${LABEL}_k${candidate_size}"

  echo
  echo "Sanity sweep candidate_size=$candidate_size"

  SUCC_DENOVO_MODEL_OUTPUT_DIR="$MODEL_OUTPUT_DIR" \
  SUCC_DENOVO_RESUME_CHECKPOINT="$CHECKPOINT" \
  SUCC_DENOVO_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_2p7p_materializer_sanity_${suffix}" \
  SUCC_DENOVO_BENCHMARK_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_2p7p_materializer_sanity_${suffix}/benchmark_ours" \
  SUCC_DENOVO_MATERIALIZED_METHODS="$METHODS" \
  SUCC_DENOVO_METHOD_LABEL="" \
  SUCC_DENOVO_PROPERTY_RERANK_CANDIDATES="$candidate_size" \
  SUCC_DENOVO_PROPERTY_RERANK_WEIGHT="$PROPERTY_RERANK_WEIGHT" \
  SUCC_DENOVO_STRICT_RERANK_WEIGHT="$STRICT_RERANK_WEIGHT" \
  SUCC_DENOVO_LATENT_RERANK_WEIGHT="$LATENT_RERANK_WEIGHT" \
  SUCC_DENOVO_RANDOM_RERANK_SEED="$RANDOM_RERANK_SEED" \
  SUCC_DENOVO_SLURM_JOB_NAME="succ-2p7p-san-${suffix}" \
  bash "$SCRIPT_DIR/submit_denovo_2p7p_ours_benchmark.sh"

  SUCC_OOD_MODEL_OUTPUT_DIR="$MODEL_OUTPUT_DIR" \
  SUCC_OOD_RESUME_CHECKPOINT="$CHECKPOINT" \
  SUCC_OOD_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_ood_materializer_sanity_${suffix}" \
  SUCC_OOD_BENCHMARK_OUTPUT_DIR="SketchMol-Understanding-Condition/outputs/denovo_ood_materializer_sanity_${suffix}/benchmark_ours" \
  SUCC_OOD_MATERIALIZED_METHODS="$METHODS" \
  SUCC_OOD_METHOD_LABEL="" \
  SUCC_OOD_PROPERTY_RERANK_CANDIDATES="$candidate_size" \
  SUCC_OOD_PROPERTY_RERANK_WEIGHT="$PROPERTY_RERANK_WEIGHT" \
  SUCC_OOD_STRICT_RERANK_WEIGHT="$STRICT_RERANK_WEIGHT" \
  SUCC_OOD_LATENT_RERANK_WEIGHT="$LATENT_RERANK_WEIGHT" \
  SUCC_OOD_RANDOM_RERANK_SEED="$RANDOM_RERANK_SEED" \
  SUCC_OOD_SLURM_JOB_NAME="succ-ood-san-${suffix}" \
  bash "$SCRIPT_DIR/submit_denovo_ood_ours_benchmark.sh"
done

echo
echo "Materializer sanity sweep submitted."
