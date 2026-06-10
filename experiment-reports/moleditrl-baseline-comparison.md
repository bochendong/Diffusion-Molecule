# MolEditRL Reference Comparison

| 字段 | 值 |
| --- | --- |
| **状态** | updated with SUCC v2 fix results |
| **最后更新** | 2026-06-09 |
| **来源** | MolEditRL Table1 screenshot |
| **用途** | 对照 SUCC / Unified × MolEdit |

## Overlap Table Metrics（Rotbonds↓ / MW↑ / SA↓）

| task | MolEditRL Acc(0.65) | SUCC v1 | SUCC v2 fix | Unified v1 |
| --- | ---: | ---: | ---: | ---: |
| Rotbonds↓ | 0.634 | 0.333 (n=3) | **0.740 (n=100)** | 0.000 (n=3) |
| MW↑ | 0.404 | 0.000 (n=1) | **0.631 (n=84)** | 0.000 (n=1) |
| SA↓ | 0.628 | 0.333 (n=15) | **0.480 (n=100)** | 0.000 (n=15) |
| **mean** | **0.555** | 0.222 | **0.617** | 0.000 |

Acc(0.15) overlap mean：MolEditRL **0.838**，SUCC v2 **0.753**，SUCC v1 0.445，Unified v1 0.111。

## Materialized 2p strict（similarity rerank）

| run | 2p strict | mean source Tani | strict@0.4 |
| --- | ---: | ---: | ---: |
| SUCC v1 | 0.450 | 0.428 | 0.379 |
| **SUCC v2 fix** | **0.727** | **0.580** | **0.729** |
| Unified v1 | ~0 (4p+) | 0.158 | 0.001 |
| MolEditRL reference | — | — | — |

## 结论

1. **SUCC v2 fix 是当前最佳线**：overlap table mean Acc(0.65) **0.617**，已超过 MolEditRL 截图 baseline 0.555。
2. **MW↑ 从 0 修复到 0.631**，Table1-balanced 训练是关键。
3. **Unified source-anchored v2** latent eval 显示 source fingerprint 0.953，但 materialized benchmark 待跑。
4. 完整 Table1（10 task）仍需 PyTDC（GSK3B/DRD2）。

详细报告：
- [SUCC v2 fix](understanding-condition/moledit-instruct-v2-fix.md)
- [Unified source-anchored v2](unified-3m/moledit-sourceanchored-v2.md)
