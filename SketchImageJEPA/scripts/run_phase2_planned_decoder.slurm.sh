#!/bin/bash
#SBATCH --job-name=sketchimage-phase2
#SBATCH --account=def-hup-ab
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=./sketchimage-phase2-%j.log
#SBATCH --error=./sketchimage-phase2-%j.log

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=(
    "/scratch/bdong/venvs/phystabmol/bin/python"
    "/home/bdong/scratch/venvs/phystabmol/bin/python"
    "/scratch/bdong/venvs/sketchimage-rdkit/bin/python"
    "$(command -v python3 2>/dev/null || true)"
  )
  for python_candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [[ -n "$python_candidate" && -x "$python_candidate" ]] && "$python_candidate" - <<'PY' >/dev/null 2>&1
import torch
PY
    then
      export SKETCHIMAGE_PYTHON_BIN="$python_candidate"
      break
    fi
  done
fi
export SKETCHIMAGE_PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-$(command -v python3)}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
echo "python=$SKETCHIMAGE_PYTHON_BIN"
echo "oracle_decoder_dir=${SKETCHIMAGE_ORACLE_DECODER_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED:-7}}"
echo "train_csv=${SKETCHIMAGE_TRAIN_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED:-7}_train.csv}"
echo "eval_csv=${SKETCHIMAGE_EVAL_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED:-7}_eval.csv}"
echo "run_name=${SKETCHIMAGE_RUN_NAME:-phase2_planned_decoder_2048_seed${SKETCHIMAGE_SEED:-7}}"

"$SKETCHIMAGE_PYTHON_BIN" -c "import sys; print('python_executable=', sys.executable)"
"$SKETCHIMAGE_PYTHON_BIN" - <<'PY'
import torch
print("torch=", torch.__version__)
print("torch_cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch_cuda_device=", torch.cuda.get_device_name(0))
PY
"$SKETCHIMAGE_PYTHON_BIN" - <<'PY'
try:
    import rdkit  # noqa: F401
    print("rdkit=available")
except Exception:
    print("rdkit=unavailable")
PY

bash scripts/run_phase2_planned_decoder.sh

echo "SketchImage-JEPA Phase 2A Slurm job finished at $(date +"%Y-%m-%dT%H:%M:%S%z")"
