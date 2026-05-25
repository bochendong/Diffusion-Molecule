#!/bin/bash
#SBATCH --job-name=sketchimage-audit
#SBATCH --account=def-hup-ab
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=./outputs/logs/sketchimage-cpu-%j.log
#SBATCH --error=./outputs/logs/sketchimage-cpu-%j.log

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p outputs/logs

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

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
echo "run_dir=${SKETCHIMAGE_RUN_DIR:-<not provided>}"
echo "train_csv=${SKETCHIMAGE_TRAIN_CSV:-<not provided>}"
echo "eval_csv=${SKETCHIMAGE_EVAL_CSV:-<not provided>}"

"$SKETCHIMAGE_PYTHON_BIN" -c "import sys; print('python_executable=', sys.executable)"
"$SKETCHIMAGE_PYTHON_BIN" - <<'PY'
try:
    import rdkit  # noqa: F401
    print("rdkit=available")
except Exception:
    print("rdkit=unavailable")
PY

bash scripts/audit_benchmark.sh

echo "SketchImage-JEPA benchmark audit job finished at $(date +"%Y-%m-%dT%H:%M:%S%z")"
