# Unified 3M × MolEdit Source-Anchored v2

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + latent eval + materialized benchmark + table metrics） |
| **最后更新** | 2026-06-10 |
| **项目** | `SketchMol-Unified-3MDiffusion` |
| **训练 Job** | `15866599`（`smu3m-srcanch-v2`） |
| **Benchmark Jobs** | `15884514`–`15884521`（5 shards）→ `15884576`（merge）→ `15884578`（table） |
| **基线** | v1: [moledit-instruct-v1.md](moledit-instruct-v1.md) |

## Slurm 运行记录

| Job | 名称 | 结果 |
| --- | --- | --- |
| `15866599` | `smu3m-srcanch-v2` | 完成（3m17s） |
| `15872825` | `smu3m-diff-bench` | **超时**（4h，无产出） |
| `15884514`–`15884521` | `smu3m-diff-bench`（5 shards） | 完成（~8h 并行） |
| `15884576` | `smu3m-bench-merge` | 完成 |
| `15884578` | `smu3m-moledit-table` | 完成（5s） |

首次单 job 4h 不够；改用 5 shard × 6h 后成功。

## 配置要点

| 变量 | 值 |
| --- | --- |
| 输出 | `outputs/unified_generation_moledit_sourceanchored_v2/` |
| `SMU3M_DIFFUSION_TARGET` | `source_residual` |
| source fingerprint prior blend | 0.90 |
| fingerprint guard / radius loss | 0.75 / 0.25 |

## Latent eval（n=1000）

| 指标 | v1 | source-anchored v2 |
| --- | ---: | ---: |
| latent MAE | 0.262 | **0.139** |
| target fingerprint cosine | 0.751 | **0.827** |
| source fingerprint cosine | — | **0.953** |
| property MAE beats source | 87.2% | **99.5%** |
| fingerprint beats source | 2.2% | **23.9%** |

## Materialized Benchmark（5-shard merge，n=1000）

`edit_latent_source_similarity_rerank`：

| 指标 | v1 | source-anchored v2 |
| --- | ---: | ---: |
| mean source Tani | 0.158 | **0.167** |
| strict@0.4 | 0.001 | 0.001 |
| 4p strict | 0.279 | 0.205 |
| 5p strict | 0.286 | **0.269** |
| 6p strict | 0.244 | **0.341** |
| 7p strict | 0.158 | **0.181** |

**Latent→materialized gap**：eval_latent 上 source fingerprint 0.953，但 retrieval materialized mean Tani 仍仅 ~0.17，与 v1 同量级。训练信号未有效传导到 candidate rerank 路径。

## MolEdit Table Metrics（job `15884578`）

| task | Acc(0.65) | Acc(0.15) | n |
| --- | ---: | ---: | ---: |
| Rotbonds↓ | 0 | 0.667 | 3 |
| MW↑ | 0 | 1.000 | 1 |
| SA↓ | 0 | 0.267 | 15 |

overlap mean Acc(0.65)=0（样本极少，且非 Table1-balanced eval）。

## 产出路径

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_sourceanchored_v2/
  eval_latent/metrics.json
  benchmark_materialized_primary_fast/
  moledit_table_metrics_sourceanchored_v2/
```

## 结论

1. **Latent 层改善明显**，但 **materialized source 保真几乎不变**（Tani 0.17 vs v1 0.16）。
2. **4p–7p strict** 有波动提升（6p 0.341 vs v1 0.244），但整体仍远低于 SUCC v2 fix（2p strict 0.727）。
3. **当前最佳线仍是 SUCC v2 fix**（见 [moledit-instruct-v2-fix.md](../understanding-condition/moledit-instruct-v2-fix.md)）。
4. 后续若继续 Unified 线，需修 latent→retrieval 解码/rerank 链路，而非仅加强 diffusion loss。
