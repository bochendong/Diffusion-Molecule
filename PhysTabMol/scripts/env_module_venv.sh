# Source from Slurm/job scripts after editing MODULE_* / VENV_DIR if needed.
# Example: source "$(dirname "$0")/env_module_venv.sh"

MODULE_GCC="${MODULE_GCC:-gcc}"
MODULE_PYTHON="${MODULE_PYTHON:-python/3.11.5}"
MODULE_RDKIT="${MODULE_RDKIT:-rdkit}"
# GPU: export MODULE_CUDA=cuda/12.6 (or see `module spider cuda`) before sourcing; leave unset on CPU-only nodes.
MODULE_CUDA="${MODULE_CUDA:-}"
VENV_DIR="${VENV_DIR:-/home/bdong/scratch/venvs/phystabmol}"
PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-/home/bdong/scratch/projects/Diffusion-Molecule/PhysTabMol}"

module load "$MODULE_GCC" "$MODULE_PYTHON" "$MODULE_RDKIT"
if [[ -n "$MODULE_CUDA" ]]; then
  module load "$MODULE_CUDA"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
cd "$PHYSTABMOL_ROOT"
