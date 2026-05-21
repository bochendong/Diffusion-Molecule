#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE_ROOT="${MOLPILOT_STAGE_ROOT:-${MOLPILOT_RUN_DIR:-}}"
if [[ -z "$STAGE_ROOT" ]]; then
  echo "ERROR: set MOLPILOT_STAGE_ROOT to an existing outputs/stages/<run> directory."
  echo "Example:"
  echo "  MOLPILOT_STAGE_ROOT=outputs/stages/molpilot_sequence_10000_20260520_015836 bash scripts/run_overnight_resample_ablation.sh"
  exit 2
fi
if [[ ! -d "$STAGE_ROOT" ]]; then
  echo "ERROR: stage root does not exist: $STAGE_ROOT"
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch was not found. Run this on the login node."
  exit 2
fi

TIME_LIMIT="${MOLPILOT_ABLATION_TIME:-08:00:00}"
PROFILE="${MOLPILOT_ABLATION_PROFILE:-overnight}"
if [[ -n "${MOLPILOT_ABLATION_MODES:-}" ]]; then
  MODES="$MOLPILOT_ABLATION_MODES"
elif [[ "$PROFILE" == "quick" ]]; then
  MODES="full_graph graph_only scaffold_library_only latent_only diffusion_only full_no_library full_wide"
else
  MODES="full_graph graph_only scaffold_library_only latent_only diffusion_only full_no_library full_wide graph_heavy scaffold_library_heavy full_heavy full_heavy_seed17"
fi
COMMON_MAX_PER_TASK="${MOLPILOT_MAX_REQUESTS_PER_TASK:-${MOLPILOT_EVAL_LIMIT:-1000}}"
COMMON_EVAL_MOLECULE_LIMIT="${MOLPILOT_EVAL_MOLECULE_LIMIT:-${MOLPILOT_LIMIT:-10000}}"
COMMON_TASKS="${MOLPILOT_EVAL_TASKS:-edit,inpaint,de_novo}"
COMMON_SAMPLES="${MOLPILOT_SAMPLES:-8}"
COMMON_DECODE_TOP_K="${MOLPILOT_DECODE_TOP_K:-4}"
COMMON_SEED="${MOLPILOT_SEED:-7}"

echo "Submitting MolPilot overnight resample ablations"
echo "  stage_root=$STAGE_ROOT"
echo "  time=$TIME_LIMIT"
echo "  profile=$PROFILE"
echo "  modes=$MODES"
echo "  eval_molecule_limit=$COMMON_EVAL_MOLECULE_LIMIT"
echo "  max_requests_per_task=$COMMON_MAX_PER_TASK"
echo "  tasks=$COMMON_TASKS"

submit_mode() {
  local mode="$1"
  export MOLPILOT_STAGE_ROOT="$STAGE_ROOT"
  export MOLPILOT_SAMPLE_SUFFIX="ablate_${mode}"
  export MOLPILOT_EVAL_MOLECULE_LIMIT="$COMMON_EVAL_MOLECULE_LIMIT"
  export MOLPILOT_MAX_REQUESTS_PER_TASK="$COMMON_MAX_PER_TASK"
  export MOLPILOT_EVAL_TASKS="$COMMON_TASKS"
  export MOLPILOT_SAMPLES="$COMMON_SAMPLES"
  export MOLPILOT_DECODE_TOP_K="$COMMON_DECODE_TOP_K"
  export MOLPILOT_SEED="$COMMON_SEED"

  unset MOLPILOT_DISABLE_SOURCE_GUIDANCE
  unset MOLPILOT_DISABLE_DIFFUSION_CANDIDATES
  unset MOLPILOT_DISABLE_LATENT_SOURCE_GUIDANCE
  unset MOLPILOT_DISABLE_GRAPH_EDITOR
  unset MOLPILOT_SOURCE_EDIT_STRENGTHS
  unset MOLPILOT_SOURCE_NEIGHBORHOOD_K
  unset MOLPILOT_GRAPH_EDIT_LIMIT
  unset MOLPILOT_SCAFFOLD_LIBRARY_K

  case "$mode" in
    full_graph)
      export MOLPILOT_SOURCE_EDIT_STRENGTHS="0.25,0.50"
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=32
      export MOLPILOT_GRAPH_EDIT_LIMIT=96
      export MOLPILOT_SCAFFOLD_LIBRARY_K=32
      ;;
    graph_only)
      export MOLPILOT_DISABLE_DIFFUSION_CANDIDATES=1
      export MOLPILOT_DISABLE_LATENT_SOURCE_GUIDANCE=1
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=0
      export MOLPILOT_GRAPH_EDIT_LIMIT=192
      export MOLPILOT_SCAFFOLD_LIBRARY_K=0
      ;;
    scaffold_library_only)
      export MOLPILOT_DISABLE_DIFFUSION_CANDIDATES=1
      export MOLPILOT_DISABLE_LATENT_SOURCE_GUIDANCE=1
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=0
      export MOLPILOT_GRAPH_EDIT_LIMIT=0
      export MOLPILOT_SCAFFOLD_LIBRARY_K=128
      ;;
    latent_only)
      export MOLPILOT_DISABLE_DIFFUSION_CANDIDATES=1
      export MOLPILOT_DISABLE_GRAPH_EDITOR=1
      export MOLPILOT_SOURCE_EDIT_STRENGTHS="0.15,0.30,0.60"
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=64
      ;;
    diffusion_only)
      export MOLPILOT_DISABLE_SOURCE_GUIDANCE=1
      ;;
    full_no_library)
      export MOLPILOT_SOURCE_EDIT_STRENGTHS="0.25,0.50"
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=32
      export MOLPILOT_GRAPH_EDIT_LIMIT=128
      export MOLPILOT_SCAFFOLD_LIBRARY_K=0
      ;;
    full_wide)
      export MOLPILOT_SOURCE_EDIT_STRENGTHS="0.15,0.30,0.60"
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=64
      export MOLPILOT_GRAPH_EDIT_LIMIT=192
      export MOLPILOT_SCAFFOLD_LIBRARY_K=64
      ;;
    graph_heavy)
      export MOLPILOT_DISABLE_DIFFUSION_CANDIDATES=1
      export MOLPILOT_DISABLE_LATENT_SOURCE_GUIDANCE=1
      export MOLPILOT_EVAL_TASKS="edit,inpaint"
      export MOLPILOT_MAX_REQUESTS_PER_TASK=0
      export MOLPILOT_SAMPLES=1
      export MOLPILOT_DECODE_TOP_K=1
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=0
      export MOLPILOT_GRAPH_EDIT_LIMIT=1024
      export MOLPILOT_SCAFFOLD_LIBRARY_K=0
      ;;
    scaffold_library_heavy)
      export MOLPILOT_DISABLE_DIFFUSION_CANDIDATES=1
      export MOLPILOT_DISABLE_LATENT_SOURCE_GUIDANCE=1
      export MOLPILOT_EVAL_TASKS="edit,inpaint"
      export MOLPILOT_MAX_REQUESTS_PER_TASK=0
      export MOLPILOT_SAMPLES=1
      export MOLPILOT_DECODE_TOP_K=1
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=0
      export MOLPILOT_GRAPH_EDIT_LIMIT=0
      export MOLPILOT_SCAFFOLD_LIBRARY_K=512
      ;;
    full_heavy)
      export MOLPILOT_EVAL_TASKS="edit,inpaint"
      export MOLPILOT_MAX_REQUESTS_PER_TASK=0
      export MOLPILOT_SAMPLES=32
      export MOLPILOT_DECODE_TOP_K=8
      export MOLPILOT_SOURCE_EDIT_STRENGTHS="0.10,0.25,0.50,0.75"
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=128
      export MOLPILOT_GRAPH_EDIT_LIMIT=512
      export MOLPILOT_SCAFFOLD_LIBRARY_K=256
      ;;
    full_heavy_seed17)
      export MOLPILOT_EVAL_TASKS="edit,inpaint"
      export MOLPILOT_MAX_REQUESTS_PER_TASK=0
      export MOLPILOT_SAMPLES=32
      export MOLPILOT_DECODE_TOP_K=8
      export MOLPILOT_SOURCE_EDIT_STRENGTHS="0.10,0.25,0.50,0.75"
      export MOLPILOT_SOURCE_NEIGHBORHOOD_K=128
      export MOLPILOT_GRAPH_EDIT_LIMIT=512
      export MOLPILOT_SCAFFOLD_LIBRARY_K=256
      export MOLPILOT_SEED=17
      ;;
    *)
      echo "ERROR: unknown ablation mode: $mode"
      exit 2
      ;;
  esac

  echo "Submitting mode=$mode suffix=$MOLPILOT_SAMPLE_SUFFIX"
  sbatch \
    --time="$TIME_LIMIT" \
    --job-name="mp-${mode:0:12}" \
    --output="./molpilot-${mode}-%j.log" \
    --error="./molpilot-${mode}-%j.log" \
    --export=ALL \
    scripts/resample_existing_stage.slurm.sh
}

for mode in $MODES; do
  submit_mode "$mode"
done

echo "Submitted all ablations. After jobs finish, run:"
echo "  MOLPILOT_STAGE_ROOT=$STAGE_ROOT bash scripts/summarize_resample_ablation.sh"
