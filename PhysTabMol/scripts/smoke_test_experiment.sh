#!/usr/bin/env bash
# PhysTabMol 冒烟测试：在 salloc / 登录节点 / Slurm 作业里验证环境与一条极简 experiment。
#
# GPU — 登录节点一键（内部会 salloc）:
#   ./scripts/smoke_test_with_salloc.sh
#
# GPU — 已有 salloc/sbatch 给的 GPU shell 时:
#   export MODULE_CUDA=cuda/12.6   # 按集群 module spider cuda 调整
#   ./scripts/smoke_test_experiment.sh
#
# 仅 CPU（sklearn 扩散，不验证 CUDA；适合登录节点）:
#   bash scripts/smoke_test_experiment.sh --cpu
#
# 登录节点默认没有 GPU：若 Default(GPU) 模式下 torch.cuda 不可用，脚本会退出并提示先用 salloc。
# 坚持在 CPU 上跑 PyTorch 扩散（很慢）:
#   PHYSTABMOL_ALLOW_CPU_TORCH=1 ./scripts/smoke_test_experiment.sh
#
# 可选环境变量:
#   PHYSTABMOL_DATA=data/molecules.csv   # 默认；不存在则用内置小分子集
#   MODULE_RDKIT=rdkit/2025.09.4

set -euo pipefail

unset LD_LIBRARY_PATH
unset PYTHONPATH

USE_CPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      USE_CPU=1
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [--cpu]"
      echo "GPU smoke expects CUDA (run inside salloc/sbatch with a GPU). On login nodes use --cpu."
      echo "Override: PHYSTABMOL_ALLOW_CPU_TORCH=1 to run torch diffusion on CPU (slow)."
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$USE_CPU" -eq 1 ]]; then
  unset MODULE_CUDA || true
else
  export MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"
fi

# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_module_venv.sh"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
echo "=== PhysTabMol smoke === host=$(hostname) use_cpu=$USE_CPU job=${SLURM_JOB_ID:-interactive}"
echo "cwd=$(pwd) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} MODULE_CUDA=${MODULE_CUDA:-<unset>}"

if [[ "$USE_CPU" -eq 0 ]]; then
  nvidia-smi || echo "(warn) nvidia-smi failed — still checking torch.cuda)"
fi

python - <<'PY'
import sys
mods = ["numpy", "pandas", "sklearn", "rdkit", "torch"]
for m in mods:
    __import__(m)
print("imports ok:", ", ".join(mods))
PY

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

if [[ "$USE_CPU" -eq 0 ]]; then
  if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    {
      echo "---------------------------------------------------------------------------"
      echo "No CUDA GPU visible on this host (torch.cuda.is_available() == False)."
      if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Note: SLURM_JOB_ID=${SLURM_JOB_ID} — on many clusters the command after salloc still runs on the submit/login host; launch GPU work with srun (see below)."
      else
        echo "This is normal on login nodes: there is no NVIDIA driver/GPU there."
      fi
      cat <<'EOF'
What to do:
  1) Real GPU smoke test — use the wrapper (runs srun on the compute node):
       ./scripts/smoke_test_with_salloc.sh

     Or after manual salloc: if hostname is still the login node, you must use srun:
       export MODULE_CUDA=cuda/12.6
       srun bash ./scripts/smoke_test_experiment.sh

  2) Quick test without GPU — use sklearn backend on the login node:
       ./scripts/smoke_test_experiment.sh --cpu

  3) Force PyTorch diffusion on CPU anyway (slow):
       PHYSTABMOL_ALLOW_CPU_TORCH=1 ./scripts/smoke_test_experiment.sh
---------------------------------------------------------------------------
EOF
    } >&2
    if [[ "${PHYSTABMOL_ALLOW_CPU_TORCH:-0}" != "1" ]]; then
      exit 1
    fi
    echo "(warn) PHYSTABMOL_ALLOW_CPU_TORCH=1 — continuing with torch on CPU"
  fi
fi

DATA="${PHYSTABMOL_DATA:-data/molecules.csv}"
DATA_ARGS=()
if [[ -f "$DATA" ]]; then
  DATA_ARGS=(--data "$DATA")
  echo "using dataset file: $DATA"
else
  echo "warn: '$DATA' not found — experiment will use built-in smoke molecules"
fi

if [[ "$USE_CPU" -eq 1 ]]; then
  python -m phystabmol.experiment \
    "${DATA_ARGS[@]}" \
    --backend sklearn \
    --run-name "smoke_cpu_${RUN_TS}" \
    --limit 400 \
    --contrastive-epochs 40 \
    --timesteps 40 \
    --noise-repeats 8 \
    --samples-per-condition 8 \
    --decode-top-k 3
else
  python -m phystabmol.experiment \
    "${DATA_ARGS[@]}" \
    --backend torch \
    --run-name "smoke_gpu_${RUN_TS}" \
    --limit 400 \
    --torch-epochs 2 \
    --contrastive-epochs 40 \
    --timesteps 40 \
    --noise-repeats 8 \
    --torch-batch-size 256 \
    --samples-per-condition 8 \
    --decode-top-k 3
fi

echo "=== smoke finished OK ==="
