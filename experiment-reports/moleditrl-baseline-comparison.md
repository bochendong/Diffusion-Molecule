# MolEditRL Reference Comparison

| 字段 | 值 |
| --- | --- |
| **状态** | updated with Table1 扩展结果（7/10 task，510 行） |
| **最后更新** | 2026-06-10 |
| **来源** | MolEditRL Table1 screenshot |
| **用途** | 对照 SUCC / Unified × MolEdit |

## SUCC v2 fix — eval 子集（job `15897034`，6/10 task，偏 eval）

| task | MolEditRL Acc(0.65) | SUCC v2 fix | MolEditRL Acc(0.15) | SUCC v2 fix |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.342 | 0.000 (n=100) | 0.514 | 0.000 |
| Rotbonds↓ | 0.634 | **0.740** | 0.830 | **0.920** |
| MW↑ | 0.404 | **0.631** | 0.856 | 0.750 |
| SA↓ | 0.628 | 0.480 | 0.828 | 0.590 |
| Haccept↓ LogP↑ | 0.316 | 1.000 (n=1) | 0.800 | 1.000 |
| DRD2↓ MW↓ SA↓ | 0.518 | 0.500 (n=10) | 0.724 | 0.800 |
| **mean over 6 tasks** | — | **0.559** | — | **0.677** |
| **overlap-3 mean (RB/MW/SA)** | **0.555** | **0.617** | **0.838** | **0.753** |

## SUCC v2 fix — Table1 扩展（job `15898540`，7/10 task，510 行）

| task | MolEditRL Acc(0.65) | SUCC Table1 ext | n | MolEditRL Acc(0.15) | SUCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.342 | 0.000 | 100 | 0.514 | 0.000 |
| Rotbonds↓ | 0.634 | 0.560 | 100 | 0.830 | 0.850 |
| MW↑ | 0.404 | 0.400 | 100 | 0.856 | 0.630 |
| SA↓ | 0.628 | 0.410 | 100 | 0.828 | 0.590 |
| QED↑ SA↓ | — | 0.000 | 1 | — | 1.000 |
| Haccept↓ LogP↑ | 0.316 | **0.778** | 9 | 0.800 | 0.778 |
| DRD2↓ MW↓ SA↓ | 0.518 | **0.700** | 100 | 0.724 | **0.930** |
| **mean over 7 tasks** | — | **0.407** | — | — | **0.540** |
| **overlap-3 mean (RB/MW/SA)** | **0.555** | **0.457** | — | **0.838** | **0.690** |

3 个 Table1 task（Haccept↓ SA↓、Haccept↓ MW↓、Haccept↑ MW↑ QED↓）在 enhanced 数据集中为 0 行，无法评估。

## Materialized 2p strict（similarity rerank）

| run | n | 2p strict | mean source Tani | strict@0.4 |
| --- | ---: | ---: | ---: | ---: |
| **SUCC v2 fix Table1 ext** | 510 | **0.778** | 0.519 | 0.592 |
| SUCC v2 fix eval 子集 | 395 | 0.727 | 0.580 | 0.729 |
| SUCC v1 | 395 | 0.450 | 0.428 | 0.379 |
| Unified src-anchored v2 | — | ~0 (4p+) | 0.167 | 0.001 |

## 结论

1. **两套数字需分开看**：eval 子集 overlap-3 Acc(0.65) **0.617** > MolEditRL **0.555**；Table1 扩展满采样 overlap-3 **0.457** < MolEditRL **0.555**。
2. **DRD2 组合 task 在满 100 行上 Acc(0.65)=0.70**，超过 MolEditRL 0.518；**GSK3B↑ 仍为 0**。
3. **RB/MW/SA 在补全 train 行后回落**，说明此前 eval 子集结果偏乐观。
4. **Materialized 2p strict 0.78**（510 行）仍是相对 v1 的实质改进。
5. **Unified source-anchored v2** materialized mean Tani ~0.17，table overlap Acc(0.65)=0。

详细报告：
- [SUCC v2 fix](understanding-condition/moledit-instruct-v2-fix.md)
- [Unified source-anchored v2](unified-3m/moledit-sourceanchored-v2.md)
