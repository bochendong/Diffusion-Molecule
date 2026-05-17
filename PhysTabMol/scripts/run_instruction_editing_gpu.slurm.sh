#!/bin/bash
#SBATCH --job-name=phystabmol-instr
#SBATCH --account=def-hup-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=./instruction-%j.log
#SBATCH --error=./instruction-%j.log

set -euo pipefail
unset LD_LIBRARY_PATH
unset PYTHONPATH
export PHYSTABMOL_SUPPRESS_RDKIT_LOGS="${PHYSTABMOL_SUPPRESS_RDKIT_LOGS:-1}"
export PHYSTABMOL_PROGRESS="${PHYSTABMOL_PROGRESS:-1}"
export PHYSTABMOL_PROGRESS_STEP="${PHYSTABMOL_PROGRESS_STEP:-5}"

# Submit from PhysTabMol repo root:
#   cd .../PhysTabMol
#   sbatch scripts/run_instruction_editing_gpu.slurm.sh

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PHYSTABMOL_ROOT="$SLURM_SUBMIT_DIR"
else
  PHYSTABMOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$PHYSTABMOL_ROOT/scripts"
cd "$PHYSTABMOL_ROOT"

export MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"
if command -v module >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/env_module_venv.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env_module_venv.sh"
fi

echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
echo "rdkit_logs_suppressed=$PHYSTABMOL_SUPPRESS_RDKIT_LOGS progress_step=${PHYSTABMOL_PROGRESS_STEP}%"
nvidia-smi || true
python3 -c "import torch; print('cuda=', torch.cuda.is_available())" || true

DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
MULTIMODAL_CONTEXT="${PHYSTABMOL_MULTIMODAL_CONTEXT:-source_reference}"
FRAGMENT_LIBRARY="${PHYSTABMOL_FRAGMENT_TRANSFORM_LIBRARY:-data/mmp_transform_library.csv}"
if [[ ! -s "$DATASET" ]]; then
  echo "Instruction dataset not found at $DATASET; building it first."
  bash scripts/build_instruction_dataset.sh
fi
if [[ "${PHYSTABMOL_DISABLE_FRAGMENT_GROWTH_DECODER:-0}" != "1" && ! -s "$FRAGMENT_LIBRARY" ]]; then
  echo "Fragment transform library not found at $FRAGMENT_LIBRARY; building it first."
  PHYSTABMOL_MMP_TRANSFORM_LIBRARY="$FRAGMENT_LIBRARY" bash scripts/build_mmp_transform_library.sh
fi

EXTRA_ARGS=()
if [[ "${PHYSTABMOL_DISABLE_SOURCE_AWARE_DECODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-source-aware-decoder)
fi
if [[ "${PHYSTABMOL_ENABLE_SOURCE_AWARE_DECODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-source-aware-decoder)
fi
if [[ "${PHYSTABMOL_DISABLE_MMP_DECODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-mmp-decoder)
fi
if [[ "${PHYSTABMOL_ENABLE_MMP_DECODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-mmp-decoder)
fi
if [[ "${PHYSTABMOL_ALLOW_TARGET_REFERENCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-target-reference)
fi
if [[ "${PHYSTABMOL_LATENT_VAE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--latent-vae)
fi
if [[ "${PHYSTABMOL_DISABLE_FRAGMENT_GROWTH_DECODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-fragment-growth-decoder)
fi
if [[ "${PHYSTABMOL_DISABLE_INSTRUCTION_GUIDED_PLAN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-instruction-guided-plan)
fi
if [[ "${PHYSTABMOL_DISABLE_INSTRUCTION_FEATURES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-instruction-features)
fi
if [[ "${PHYSTABMOL_BLIND_INSTRUCTION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--blind-instruction)
fi
if [[ "$MULTIMODAL_CONTEXT" == "source_reference" || "$MULTIMODAL_CONTEXT" == "full" ]]; then
  if ! head -n 1 "$DATASET" | tr ',' '\n' | grep -qx 'reference_smiles'; then
    echo "Dataset $DATASET has no reference_smiles column; rebuilding for multimodal source_reference/full."
    bash scripts/build_instruction_dataset.sh
  fi
fi

python3 -m phystabmol.instruction_experiment \
  --dataset "$DATASET" \
  --backend "${PHYSTABMOL_BACKEND:-torch}" \
  --run-name "${PHYSTABMOL_RUN_NAME:-instruction_slurm_${SLURM_JOB_ID:-manual}}" \
  --limit "${PHYSTABMOL_LIMIT:-0}" \
  --split-column "${PHYSTABMOL_SPLIT_COLUMN:-split}" \
  --planner-mode "${PHYSTABMOL_PLANNER_MODE:-diffusion}" \
  --eval-limit "${PHYSTABMOL_EVAL_LIMIT:-2000}" \
  --samples-per-instruction "${PHYSTABMOL_SAMPLES:-8}" \
  --decode-top-k "${PHYSTABMOL_DECODE_TOP_K:-2}" \
  --multimodal-context "$MULTIMODAL_CONTEXT" \
  --mmp-pool-size "${PHYSTABMOL_MMP_POOL_SIZE:-768}" \
  --mmp-source-neighbors "${PHYSTABMOL_MMP_SOURCE_NEIGHBORS:-256}" \
  --mmp-delta-neighbors "${PHYSTABMOL_MMP_DELTA_NEIGHBORS:-256}" \
  --mmp-tag-neighbors "${PHYSTABMOL_MMP_TAG_NEIGHBORS:-384}" \
  --mmp-reference-neighbors "${PHYSTABMOL_MMP_REFERENCE_NEIGHBORS:-128}" \
  --mmp-verify-candidates "${PHYSTABMOL_MMP_VERIFY_CANDIDATES:-512}" \
  --source-aware-pool-size "${PHYSTABMOL_SOURCE_AWARE_POOL_SIZE:-512}" \
  --source-aware-verify-candidates "${PHYSTABMOL_SOURCE_AWARE_VERIFY_CANDIDATES:-384}" \
  --fragment-transform-library "$FRAGMENT_LIBRARY" \
  --fragment-pair-neighbors "${PHYSTABMOL_FRAGMENT_PAIR_NEIGHBORS:-0}" \
  --fragment-neighbors "${PHYSTABMOL_FRAGMENT_NEIGHBORS:-16}" \
  --fragment-attachment-limit "${PHYSTABMOL_FRAGMENT_ATTACHMENT_LIMIT:-3}" \
  --fragment-growth-steps "${PHYSTABMOL_FRAGMENT_GROWTH_STEPS:-2}" \
  --fragment-growth-beam-size "${PHYSTABMOL_FRAGMENT_GROWTH_BEAM_SIZE:-12}" \
  --fragment-second-step-neighbors "${PHYSTABMOL_FRAGMENT_SECOND_STEP_NEIGHBORS:-6}" \
  --fragment-growth-mw-gap "${PHYSTABMOL_FRAGMENT_GROWTH_MW_GAP:-25.0}" \
  --fragment-bonus "${PHYSTABMOL_FRAGMENT_BONUS:-0.24}" \
  --fragment-exact-penalty "${PHYSTABMOL_FRAGMENT_EXACT_PENALTY:-0.30}" \
  --fragment-prompt-match-bonus "${PHYSTABMOL_FRAGMENT_PROMPT_MATCH_BONUS:-3.0}" \
  --fragment-prompt-miss-penalty "${PHYSTABMOL_FRAGMENT_PROMPT_MISS_PENALTY:-10.0}" \
  --torch-epochs "${PHYSTABMOL_TORCH_EPOCHS:-80}" \
  --torch-batch-size "${PHYSTABMOL_TORCH_BATCH_SIZE:-1024}" \
  --torch-hidden-dim "${PHYSTABMOL_TORCH_HIDDEN_DIM:-1024}" \
  --torch-layers "${PHYSTABMOL_TORCH_LAYERS:-6}" \
  --vae-latent-dim "${PHYSTABMOL_VAE_LATENT_DIM:-16}" \
  --vae-hidden-dim "${PHYSTABMOL_VAE_HIDDEN_DIM:-512}" \
  --vae-layers "${PHYSTABMOL_VAE_LAYERS:-3}" \
  --vae-epochs "${PHYSTABMOL_VAE_EPOCHS:-60}" \
  --vae-batch-size "${PHYSTABMOL_VAE_BATCH_SIZE:-1024}" \
  --vae-lr "${PHYSTABMOL_VAE_LR:-0.001}" \
  --vae-beta "${PHYSTABMOL_VAE_BETA:-0.001}" \
  --source-anchor-weight "${PHYSTABMOL_SOURCE_ANCHOR_WEIGHT:-0.35}" \
  --source-count-anchor-weight "${PHYSTABMOL_SOURCE_COUNT_ANCHOR_WEIGHT:-0.15}" \
  --source-anchor-neighbors "${PHYSTABMOL_SOURCE_ANCHOR_NEIGHBORS:-32}" \
  --timesteps "${PHYSTABMOL_TIMESTEPS:-80}" \
  --noise-repeats "${PHYSTABMOL_NOISE_REPEATS:-8}" \
  "${EXTRA_ARGS[@]}"

echo "Instruction editing job finished at $(date -Is)"
