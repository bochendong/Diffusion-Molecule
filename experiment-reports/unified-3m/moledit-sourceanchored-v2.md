# Unified 3M × MolEdit Source-Anchored v2

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + latent eval）；materialized benchmark **待跑** |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Unified-3MDiffusion` |
| **训练 Job** | `15866599`（`smu3m-srcanch-v2`） |
| **基线** | v1: [moledit-instruct-v1.md](moledit-instruct-v1.md) |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15866599` | `smu3m-srcanch-v2` | **3m17s** | 0 |

economy profile（batch=512，5 epochs，H100 20GB MIG）。

## 目的

在 v1 基础上改用 **source-residual diffusion** + source-anchor clamp，解决 v1 materialized benchmark 中 mean source Tanimoto 仅 0.158 的问题。

## 配置要点

| 变量 | 值 |
| --- | --- |
| 输出 | `outputs/unified_generation_moledit_sourceanchored_v2/` |
| `SMU3M_DIFFUSION_TARGET` | `source_residual` |
| `SMU3M_SOURCE_FINGERPRINT_PRIOR_BLEND` | 0.90 |
| `SMU3M_FINGERPRINT_GUARD_LOSS_WEIGHT` | 0.75 |
| `SMU3M_SOURCE_RADIUS_LOSS_WEIGHT` | 0.25 |
| `SMU3M_SOURCE_RESIDUAL_RADIUS_MULTIPLIER` | 1.25 |

## 结果摘要（eval_latent，n=1000）

| 指标 | v1 | source-anchored v2 |
| --- | ---: | ---: |
| latent MAE | 0.262 | **0.139** |
| target fingerprint cosine | 0.751 | **0.827** |
| **source fingerprint cosine** | ~0.84 (pair) | **0.953** |
| property MAE beats source | 87.2% | **99.5%** |
| fingerprint beats source | 2.2% | **23.9%** |
| source_residual_clamped_rate | — | 6.2% |

Source 保真显著改善：source→generated fingerprint cosine 0.953 vs v1 materialized mean Tani 0.158。

## 尚未完成

Materialized benchmark + MolEdit table metrics 未自动链式提交。训练完成后运行：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_sourceanchored_v2_benchmark.sh
```

## 产出路径

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_sourceanchored_v2/
  latent_diffusion/latent_diffusion_generation.pt
  eval_latent/metrics.json
  eval_latent/predictions.csv
```

日志：`SketchMol-Unified-3MDiffusion/logs/smu3m-srcanch-v2-15866599.log`

## 结论与下一步

1. **Latent eval 大幅改善**：source fingerprint cosine 0.953，property beats source 99.5%。
2. **待 materialized benchmark** 验证 strict@Tanimoto 是否从 v1 的 0.001 提升。
3. **对照 SUCC v2 fix**：SUCC table overlap mean Acc(0.65)=0.617 已接近 MolEditRL；Unified 需 benchmark 后才能对比。
