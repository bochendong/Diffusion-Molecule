# MolEditRL Reference Comparison

| 字段 | 值 |
| --- | --- |
| **状态** | reference baseline recorded from MolEditRL Table1 screenshot |
| **最后更新** | 2026-06-09 |
| **来源** | `/Users/dongpochen/Desktop/截屏2026-06-09 下午3.18.55.png` |
| **用途** | 对照 SUCC UniVideo x MolEdit 与 Unified 3M x MolEdit |

## MolEditRL Table1 Baseline

截图中 MolEditRL 行如下。箭头表示目标性质上升或下降。这里先记录
`Acc_all` / `Acc_valid` / `Validity` / `FCD`，方便后续和我们自己的
MolEdit table metrics 对齐。

| task | Validity | Acc_all(0.65) | Acc_valid(0.65) | Acc_all(0.15) | Acc_valid(0.15) | FCD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSK3beta↑ | 0.952 | 0.342 | 0.3592 | 0.514 | 0.5399 | 7.9953 |
| Rotbonds↓ | 0.984 | 0.634 | 0.6443 | 0.830 | 0.8435 | 6.9724 |
| MW↑ | 0.960 | 0.404 | 0.4208 | 0.856 | 0.8917 | 6.5879 |
| SA↓ | 0.988 | 0.628 | 0.6356 | 0.828 | 0.8381 | 7.1027 |
| Haccept↓ + SA↓ | 0.972 | 0.346 | 0.3560 | 0.510 | 0.5247 | 11.3769 |
| QED↑ + SA↓ | 0.974 | 0.632 | 0.6489 | 0.788 | 0.8090 | 7.5393 |
| Haccept↓ + LogP↑ | 0.946 | 0.316 | 0.3340 | 0.800 | 0.8457 | 10.1124 |
| Haccept↓ + MW↓ | 0.942 | 0.252 | 0.2675 | 0.660 | 0.7006 | 11.3373 |
| DRD2↓ + MW↓ + SA↓ | 0.986 | 0.518 | 0.5254 | 0.724 | 0.7343 | 7.2758 |
| Haccept↑ + MW↑ + QED↓ | 0.958 | 0.430 | 0.4489 | 0.756 | 0.7891 | 9.7922 |

## Overlap With Our Current Runs

当前我们自己的 table metrics 只覆盖 3 个可本地计算的任务：
Rotbonds↓、MW↑、SA↓。GSK3beta / DRD2 需要 PyTDC oracle，目前跳过；
多目标组合任务也还没有完整覆盖。因此下面只是 **overlap sanity
comparison**，不是完整 Table1 复现。

| task | MolEditRL Acc_all(0.65) | SUCC Acc_all(0.65) | Unified 3M Acc_all(0.65) | MolEditRL Acc_all(0.15) | SUCC Acc_all(0.15) | Unified 3M Acc_all(0.15) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rotbonds↓ | 0.634 | 0.333 | 0.000 | 0.830 | 0.667 | 0.000 |
| MW↑ | 0.404 | 0.000 | 0.000 | 0.856 | 0.000 | 0.000 |
| SA↓ | 0.628 | 0.333 | 0.000 | 0.828 | 0.667 | 0.333 |
| **mean over overlap** | **0.555** | **0.222** | **0.000** | **0.838** | **0.445** | **0.111** |

## Interpretation

1. **MolEditRL is still clearly ahead on table metrics.** On the 3 overlapping
   tasks, MolEditRL averages 0.555 Acc_all(0.65) and 0.838 Acc_all(0.15).
   SUCC reaches 0.222 / 0.445, while Unified 3M reaches 0.000 / 0.111.
2. **SUCC is meaningfully better than Unified 3M in the current materialized
   table metric path.** SUCC gets partial credit on Rotbonds↓ and SA↓; Unified
   3M mostly fails the strict table metric despite having latent-level property
   signal.
3. **MW↑ is the most obvious current failure case.** Both SUCC and Unified 3M
   score 0 on MW↑ in the current sampled table run.
4. **Caveat: our overlap counts are tiny.** Current table jobs report
   Rotbonds↓ n=3, MW↑ n=1, SA↓ n=15, so the numbers are useful for direction
   but not yet a stable paper-level comparison. A full comparison needs PyTDC
   installed and a larger Table1-balanced eval pass.

## Relation To Materialized Benchmarks

The materialized benchmark tells a slightly different story:

| run | stronger metric | source similarity issue |
| --- | --- | --- |
| SUCC UniVideo x MolEdit | `edit_latent_source_similarity_rerank` has 2p strict 0.450 and mean source Tani 0.428 | better source/edit balance than Unified, but still below MolEditRL table metrics |
| Unified 3M x MolEdit | latent eval has property signal: property MAE beats source 87.2% | materialized mean source Tani is only 0.158, so generated candidates drift far from source |

Practical conclusion: **our best current line is SUCC**, not Unified 3M. It is
closer to a source-conditioned edit behavior, but it is still substantially
behind MolEditRL on the official-style table metrics.
