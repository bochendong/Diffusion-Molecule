# MolEditRL Reference Comparison

| 字段 | 值 |
| --- | --- |
| **状态** | updated with PyTDC-enabled SUCC v2 table metrics |
| **最后更新** | 2026-06-10 |
| **来源** | MolEditRL Table1 screenshot |
| **用途** | 对照 SUCC / Unified × MolEdit |

## SUCC v2 fix Table Metrics（job `15897034`，6/10 Table1 tasks）

| task | MolEditRL Acc(0.65) | SUCC v2 fix | MolEditRL Acc(0.15) | SUCC v2 fix |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.342 | 0.000 (n=100) | 0.514 | 0.000 |
| Rotbonds↓ | 0.634 | **0.740** | 0.830 | **0.920** |
| MW↑ | 0.404 | **0.631** | 0.856 | 0.750 |
| SA↓ | 0.628 | 0.480 | 0.828 | 0.590 |
| Haccept↓ LogP↑ | 0.316 | 1.000 (n=1) | 0.800 | 1.000 |
| DRD2↓ MW↓ SA↓ | 0.518 | **0.500** (n=10) | 0.724 | 0.800 |
| **mean over 6 tasks** | — | **0.559** | — | **0.677** |
| **overlap-3 mean (RB/MW/SA)** | **0.555** | **0.617** | **0.838** | **0.753** |

## Materialized 2p strict（similarity rerank）

| run | 2p strict | mean source Tani | strict@0.4 |
| --- | ---: | ---: | ---: |
| **SUCC v2 fix** | **0.727** | **0.580** | **0.729** |
| SUCC v1 | 0.450 | 0.428 | 0.379 |
| Unified src-anchored v2 | ~0 (4p+) | 0.167 | 0.001 |
| Unified v1 | ~0 (4p+) | 0.158 | 0.001 |

## 结论

1. **SUCC v2 fix 仍是当前最佳线**；overlap-3 Acc(0.65) **0.617** > MolEditRL **0.555**。
2. **GSK3B↑ 是当前明显短板**（0 vs MolEditRL 0.342）；DRD2 组合 task 已可评估（0.5）。
3. **PyTDC 已在集群跑通**（`setup_moledit_tdc.sh` + `rdkit.six` shim）；benchmark 预测仍只覆盖 6/10 Table1 task。
4. **Unified source-anchored v2** materialized mean Tani ~0.17，table overlap Acc(0.65)=0。

详细报告：
- [SUCC v2 fix](understanding-condition/moledit-instruct-v2-fix.md)
- [Unified source-anchored v2](unified-3m/moledit-sourceanchored-v2.md)
