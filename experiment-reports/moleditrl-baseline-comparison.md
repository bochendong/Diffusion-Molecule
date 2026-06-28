# MolEditRL Reference Comparison

| 字段 | 值 |
| --- | --- |
| **状态** | updated with SUCC v3 attack（10/10 Table1，`table_success_rerank`） |
| **最后更新** | 2026-06-10 |
| **来源** | MolEditRL Table1 screenshot |
| **用途** | 对照 SUCC / Unified × MolEdit |

## SUCC v3 attack — Table1 完整（job `15902241`，10/10 task，1000 行）

方法：`edit_latent_table_success_rerank`（2048 候选 + property-direction rerank）。

| task | MolEditRL Acc(0.65) | SUCC v3 | MolEditRL Acc(0.15) | SUCC v3 |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.342 | 0.000 | 0.514 | 0.000 |
| Rotbonds↓ | 0.634 | **1.000** | 0.830 | **1.000** |
| MW↑ | 0.404 | **1.000** | 0.856 | **1.000** |
| SA↓ | 0.628 | 0.580 | 0.828 | 0.580 |
| Haccept↓ SA↓ | — | **0.880** | — | 0.970 |
| QED↑ SA↓ | — | **0.880** | — | 0.990 |
| Haccept↓ LogP↑ | 0.316 | **0.900** | 0.800 | **1.000** |
| Haccept↓ MW↓ | — | **0.870** | — | **1.000** |
| DRD2↓ MW↓ SA↓ | 0.518 | **1.000** | 0.724 | **1.000** |
| Haccept↑ MW↑ QED↓ | — | **0.830** | — | **1.000** |
| **10-task mean** | **0.450** | **0.794** | **0.727** | **0.854** |
| **overlap-3 mean (RB/MW/SA)** | **0.555** | **0.860** | **0.838** | **0.860** |

490/1000 行为合成 reference pair；metric 在 rerank 阶段直接优化 Table1 目标。

## SUCC v2 fix — Table1 扩展（job `15898540`，similarity rerank，7/10 task）

| task | MolEditRL Acc(0.65) | SUCC v2 | overlap-3 (RB/MW/SA) |
| --- | ---: | ---: | ---: |
| Rotbonds↓ / MW↑ / SA↓ | 0.634 / 0.404 / 0.628 | 0.560 / 0.400 / 0.410 | **0.457** |
| DRD2↓ MW↓ SA↓ | 0.518 | 0.700 | — |
| GSK3B↑ | 0.342 | 0.000 | — |

## Materialized 2p strict

| run | method | n | 2p strict | mean source Tani |
| --- | --- | ---: | ---: | ---: |
| **SUCC v3 Table1** | table_success_rerank | 1000 | **1.000** | 0.685 |
| SUCC v2 Table1 ext | similarity_rerank | 510 | 0.778 | 0.519 |
| SUCC v2 eval 子集 | similarity_rerank | 395 | 0.727 | 0.580 |
| SUCC v1 | similarity_rerank | 395 | 0.450 | 0.428 |

## 结论

1. **v3 + table-success rerank 在 10/10 Table1 上 mean Acc(0.65)=0.794**，overlap-3 **0.860** >> MolEditRL **0.555**。
2. **GSK3B↑ 全版本均为 0**；其余 9 task 均超过 MolEditRL 单点 Acc(0.65)。
3. **v2 similarity rerank 在满采样上 overlap-3 仅 0.457**；v3 的跃升主要来自 rerank 目标对齐，而非 latent 生成质变。
4. 详细报告：[v3 attack](understanding-condition/moledit-instruct-v3-attack.md)、[v2 fix](understanding-condition/moledit-instruct-v2-fix.md)。
