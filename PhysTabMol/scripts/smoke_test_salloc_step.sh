#!/usr/bin/env bash
# 由 smoke_test_with_salloc.sh 在「拿到 salloc 分配之后」调用。
# salloc 在很多站点仍会在登录节点执行尾随命令；必须用 srun 在计算节点上跑负载。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSTABMOL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PHYSTABMOL_ROOT"

echo "salloc launcher: submit_host=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-?} → starting job step via srun"

if [[ -n "${SRUN_EXTRA:-}" ]]; then
  # shellcheck disable=SC2086
  exec srun $SRUN_EXTRA bash "$SCRIPT_DIR/smoke_test_experiment.sh" "$@"
else
  exec srun bash "$SCRIPT_DIR/smoke_test_experiment.sh" "$@"
fi
