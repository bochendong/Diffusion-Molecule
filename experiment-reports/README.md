# 实验报告

本目录存放各实验的可更新报告（Markdown）。

## 当前实验索引

| 文件夹 | 报告文件 | 说明 |
| --- | --- | --- |
| `.` | [moleditrl-baseline-comparison.md](moleditrl-baseline-comparison.md) | MolEditRL vs SUCC v1/v2/v3 / Unified 对照 |
| `unified-3m/` | [moledit-instruct-v1.md](unified-3m/moledit-instruct-v1.md) | Unified 3M MolEdit v1（`15810199`/`15837897`） |
| `unified-3m/` | [moledit-sourceanchored-v2.md](unified-3m/moledit-sourceanchored-v2.md) | **完成** source-anchored v2（`15866599`/`15884514-21`/`15884578`） |
| `understanding-condition/` | [moledit-instruct-v1.md](understanding-condition/moledit-instruct-v1.md) | SUCC MolEdit v1（`15838733`/`15838734`） |
| `understanding-condition/` | [moledit-instruct-v2-fix.md](understanding-condition/moledit-instruct-v2-fix.md) | **完成** SUCC v2 fix（`15866598`/`15866600`/`15897034`） |
| `understanding-condition/` | [moledit-instruct-v3-attack.md](understanding-condition/moledit-instruct-v3-attack.md) | **完成** SUCC v3 attack（`15901616`/`15902241`，10/10 Table1） |
| `understanding-condition/` | [moledit-instruct-dualmode-v1.md](understanding-condition/moledit-instruct-dualmode-v1.md) | dual-mode v1（`15963804`/`16006990`/`16006991`） |
| `understanding-condition/` | [moledit-instruct-dualmode-v2-guarded.md](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) | dual-mode v2 guarded（`16018414` 链；guard failed） |
| `understanding-condition/` | [moledit-instruct-dualmode-v3-guarded.md](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) | dual-mode v3 guarded（`16027758` 链；guard passed） |
| `understanding-condition/` | [moledit-instruct-dualmode-v4-warmstart-v1.md](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) | **完成** dual-mode v4 warmstart（`16042564` 链；guard pass，strict 未达目标） |
| `understanding-condition/` | [materializer-hybrid-sweep.md](understanding-condition/materializer-hybrid-sweep.md) | **完成** zero-source hybrid materializer sweep（`16056226`–`16056231`） |
| `understanding-condition/` | [materializer-random-sanity-sweep.md](understanding-condition/materializer-random-sanity-sweep.md) | 待运行 zero-source random shortlist sanity sweep |
| `understanding-condition/` | [denovo-2p7p-benchmark.md](understanding-condition/denovo-2p7p-benchmark.md) | **完成** de novo 2p–7p（baseline / v2_fix / dualmode） |
| `understanding-condition/` | [denovo-ood-benchmark.md](understanding-condition/denovo-ood-benchmark.md) | **完成** de novo OOD（v2_fix + dualmode） |
| `understanding-condition/` | [source-neighbor-v2-residual-ink.md](understanding-condition/source-neighbor-v2-residual-ink.md) | source-neighbor 对照（`15821981`/`15821983`） |

## 近期 Slurm Job 一览（2026-06-15 hybrid 默认验证）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16072912` | `succ-denovo-2p7p-ours`（standalone hybrid 默认） | 完成（18m49s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `16072913` | `succ-denovo-ood-ours`（standalone hybrid 默认） | 完成（12m42s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |

与 materializer sweep `16056226`/`16056227` 数值一致，确认默认切换可复现。

## 近期 Slurm Job 一览（2026-06-14 materializer sweep）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16056226` | `succ-2p7p-mat-v1` | 完成（1h26m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056227` | `succ-ood-mat-v1` | 完成（25m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056228` | `succ-2p7p-mat-v3` | 完成（1h27m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056229` | `succ-ood-mat-v3` | 完成（25m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056230` | `succ-2p7p-mat-v4` | 完成（1h26m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056231` | `succ-ood-mat-v4` | 完成（24m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |

## 近期 Slurm Job 一览（2026-06-14 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16042564` | `succ-univideo-mol`（dualmode v4 warmstart） | 完成（52m00s） | [dualmode v4](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) |
| `16043202`–`16043204` | Table1 eval/bench/table | 完成 | [dualmode v4](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) |
| `16043205` | `succ-table1-guard` | **Passed** | [dualmode v4](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) |
| `16043206` | `succ-denovo-ood-ours`（v4 ckpt） | 完成（12m06s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16043207` | `succ-denovo-2p7p-ours`（v4 ckpt） | 完成（13m48s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-14）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16027758` | `succ-univideo-mol`（dualmode v3 训练） | 完成（54m08s） | [dualmode v3](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) |
| `16028792`–`16028794` | Table1 eval/bench/table | 完成 | [dualmode v3](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) |
| `16028795` | `succ-table1-guard` | **Passed**（Acc 0.794 ≥ 0.794） | [dualmode v3](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) |
| `16028796` | `succ-denovo-ood-ours`（v3 ckpt） | 完成（12m52s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16028797` | `succ-denovo-2p7p-ours`（v3 ckpt） | 完成（13m51s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-13 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16018414` | `succ-univideo-mol`（dualmode v2 训练） | 完成（53m14s） | [dualmode v2](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) |
| `16019240`–`16019242` | Table1 eval/bench/table | 完成 | [dualmode v2](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) |
| `16019243` | `succ-table1-guard` | **Failed**（Acc 0.794 < 0.894） | [dualmode v2](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) |
| `16019244` | `succ-denovo-ood-ours`（v2 ckpt） | 完成（15m05s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16019245` | `succ-denovo-2p7p-ours`（v2 ckpt） | 完成（14m20s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-13）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16006990` | `succ-denovo-ood-dualmode` | 完成（16m09s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16006991` | `succ-denovo-2p7p-dualmode` | 完成（17m55s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-12）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15963804` | `succ-dualmode-v1`（MolEdit+de novo+OOD 训练） | 完成（54m29s） | [dualmode v1](understanding-condition/moledit-instruct-dualmode-v1.md) |

## 近期 Slurm Job 一览（2026-06-11）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15958481` | `succ-denovo-2p7p`（baseline） | 完成（1h08m） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `15959362` | `succ-denovo-2p7p-ours` | 完成（13m46s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `15959891` | `succ-denovo-ood-ours` | 完成（11m51s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |

## 历史 Slurm Job（2026-06-10）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15901616`–`15901618` | v3 attack 训练/bench/table | 完成 | [v3 attack](understanding-condition/moledit-instruct-v3-attack.md) |
| `15902238`–`15902241` | v3 Table1 扩展 eval/bench/table | 完成 | [v3 attack](understanding-condition/moledit-instruct-v3-attack.md) |
| `15898378`–`15898540` | v2 Table1 扩展 | 完成 | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |

## 历史 Slurm Job（2026-06-09 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15866598` | `succ-univideo-mol`（v2 fix） | 完成（27m16s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866600` | `succ-univideo-bench`（v2 fix） | 完成（2m37s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866601` | `succ-moledit-table`（v2 fix，缺 TDC） | 完成 | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15897034` | `succ-moledit-table`（v2 fix，PyTDC） | 完成 | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866599` | `smu3m-srcanch-v2` | 完成（3m17s） | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |
| `15884514`–`15884521` | `smu3m-diff-bench`（5 shards） | 完成 | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |
| `15884578` | `smu3m-moledit-table` | 完成 | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |

新增实验时：在对应子文件夹下新建 `{实验名}.md`，并在此表加一行链接。
