#!/usr/bin/env bash
# 在登录节点一键：salloc 申请 GPU → 在计算节点上跑 smoke_test_experiment.sh
#
#   cd .../PhysTabMol
#   ./scripts/smoke_test_with_salloc.sh
#   ./scripts/smoke_test_with_salloc.sh --cpu    # 仍会要 GPU 分区资源；只想 CPU 请直接 smoke_test_experiment.sh --cpu
#
# 默认仅满足 smoke_test_experiment.sh（小数据、极少 epoch），时间短、CPU/内存请求偏小，便于排队。
# 正式训练请用 scripts/run_phystabmol_gpu.slurm.sh，或自行 export 更大的 SALLOC_*。
#
# 常用环境变量（与 Slurm / 集群策略对齐后再提交）:
#   SALLOC_ACCOUNT=rrg-hup
#   SALLOC_GPUS=nvidia_h100_80gb_hbm3_1g.10gb:1   # Nibi 10GB MIG；整卡示例: h100:1
#   SALLOC_CPUS=4                 # 要加大再 export（例如 16）
#   SALLOC_MEM_PER_CPU=2G
#   SALLOC_TIME=0:30:00
#   SALLOC_PARTITION=...          # 可选
#   SALLOC_MAIL_USER=you@mail     # 可选，分配开始时发邮件
#   MODULE_CUDA=cuda/12.6
#   SRUN_EXTRA="--gpu-bind=closest"   # 可选，传给 srun 的额外参数（空格分隔）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSTABMOL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${SALLOC_ACCOUNT:=rrg-hup}"
: "${SALLOC_GPUS:=nvidia_h100_80gb_hbm3_1g.10gb:1}"
: "${SALLOC_CPUS:=4}"
: "${SALLOC_MEM_PER_CPU:=2G}"
: "${SALLOC_TIME:=0:30:00}"
export MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"

SALLOC_EXTRA=()
if [[ -n "${SALLOC_PARTITION:-}" ]]; then
  SALLOC_EXTRA+=(--partition="$SALLOC_PARTITION")
fi
if [[ -n "${SALLOC_MAIL_USER:-}" ]]; then
  SALLOC_EXTRA+=(--mail-user="$SALLOC_MAIL_USER" --mail-type="${SALLOC_MAIL_TYPE:-BEGIN}")
fi

cd "$PHYSTABMOL_ROOT"

echo "Submitting interactive allocation (salloc) then running GPU smoke test..."
echo "  account=$SALLOC_ACCOUNT gpus=$SALLOC_GPUS cpus=$SALLOC_CPUS mem/cpu=$SALLOC_MEM_PER_CPU time=$SALLOC_TIME"

exec salloc \
  --account="$SALLOC_ACCOUNT" \
  --gpus="$SALLOC_GPUS" \
  --cpus-per-task="$SALLOC_CPUS" \
  --mem-per-cpu="$SALLOC_MEM_PER_CPU" \
  --time="$SALLOC_TIME" \
  "${SALLOC_EXTRA[@]}" \
  -- \
  bash "$SCRIPT_DIR/smoke_test_salloc_step.sh" "$@"
