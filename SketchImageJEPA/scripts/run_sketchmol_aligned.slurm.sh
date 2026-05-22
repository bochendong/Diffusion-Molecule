#!/bin/bash
#SBATCH --job-name=sketchimage-jepa
#SBATCH --account=def-hup-ab
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=./sketchimage-jepa-%j.log
#SBATCH --error=./sketchimage-jepa-%j.log

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    export SKETCHIMAGE_PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    export SKETCHIMAGE_PYTHON_BIN="$(command -v python3)"
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "jobid=${SLURM_JOB_ID:-manual} node=$(hostname) cwd=$(pwd)"
echo "python=$SKETCHIMAGE_PYTHON_BIN"
echo "molecule_csv=${SKETCHIMAGE_MOLECULE_CSV:-<not provided>}"
echo "dataset_csv=${SKETCHIMAGE_DATASET_CSV:-data/example_tasks.csv}"
echo "run_name=${SKETCHIMAGE_RUN_NAME:-<timestamped>}"

"$SKETCHIMAGE_PYTHON_BIN" -c "import sys; print('python_executable=', sys.executable)"
"$SKETCHIMAGE_PYTHON_BIN" -c "import numpy; print('numpy=', numpy.__version__)"
"$SKETCHIMAGE_PYTHON_BIN" - <<'PY'
try:
    import rdkit  # noqa: F401
    print("rdkit=available")
except Exception:
    print("rdkit=unavailable")
PY

bash scripts/run_sketchmol_aligned.sh

echo "SketchImage-JEPA Slurm job finished at $(date +"%Y-%m-%dT%H:%M:%S%z")"
