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
| `understanding-condition/` | [materializer-random-sanity-sweep.md](understanding-condition/materializer-random-sanity-sweep.md) | **完成** random shortlist sanity sweep（`16075242`–`16075253`） |
| `understanding-condition/` | [direct-smiles-denovo-v0.md](understanding-condition/direct-smiles-denovo-v0.md) | **完成** direct SMILES de novo v0（`16079256`/`16079257`；strict≈0，mode collapse） |
| `understanding-condition/` | [direct-smiles-denovo-v1-sampled-rerank.md](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) | **完成** direct SMILES v1（best SFT n=256 **56.2%**；DPO v1 **52.1%** 未提升；RL **23.2%** collapse） |
| `understanding-condition/` | [direct-smiles-denovo-v2-mixed-condition.md](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) | **完成** direct SMILES v2 mixed condition（2p7p n=64 **68.1%**；OOD **53.1%**，7p bucket **51%**） |
| `understanding-condition/` | [denovo-2p7p-benchmark.md](understanding-condition/denovo-2p7p-benchmark.md) | **完成** de novo 2p–7p（baseline / v2_fix / dualmode） |
| `understanding-condition/` | [denovo-ood-benchmark.md](understanding-condition/denovo-ood-benchmark.md) | **完成** de novo OOD（v2_fix + dualmode） |
| `understanding-condition/` | [source-neighbor-v2-residual-ink.md](understanding-condition/source-neighbor-v2-residual-ink.md) | source-neighbor 对照（`15821981`/`15821983`） |

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v0）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16079256` | `succ-direct-smiles-2p7p` | 完成（9m51s） | [direct SMILES v0](understanding-condition/direct-smiles-denovo-v0.md) |
| `16079257` | `succ-direct-smiles-ood` | 完成（4m19s） | [direct SMILES v0](understanding-condition/direct-smiles-denovo-v0.md) |

2p7p strict **0.07%**（6000 行仅 1 个 unique SMILES）；OOD strict **0%**，validity **9.4%**。

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v1）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16079726` | `succ-direct-smiles-2p7p`（v1 sampled rerank） | 完成（1h50m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16079727` | `succ-direct-smiles-ood`（v1 sampled rerank） | 完成（23m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

v1 打破 mode collapse：2p7p strict **24.4%**（unique 5967/6000）；OOD strict **39%**（validity 95.4%）。

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v1 n=64）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16104381` | `succ-direct-smiles-2p7p-n64` | 完成（3h21m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16104383` | `succ-direct-smiles-ood-n64` | 完成（38m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

n=64 推理-only（复用 v1 ckpt）：2p7p **35.4%**（+11pp vs n=32）；OOD **54.5%**（+15.5pp）。

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v1 n=128）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16143574` | `succ-direct-smiles-2p7p-n128` | 完成（6h35m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16143575` | `succ-direct-smiles-ood-n128` | 完成（1h17m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

n=128 推理-only：2p7p **45.9%**（2p **85%** > SketchMol ref）；OOD **70.9%**。

## 近期 Slurm Job 一览（2026-06-17 direct SMILES v1 n=256）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16245062` | `succ-direct-smiles-2p7p-n256` | 完成（13h45m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16245063` | `succ-direct-smiles-ood-n256` | 完成（2h35m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

n=256 推理-only（time=20h/8h）：2p7p **56.2%**（+10.3pp vs n=128）；OOD **80.1%**（+9.2pp）。

## 近期 Slurm Job 一览（2026-06-18~19 direct SMILES 并行 fast2 + RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16303776` | `succ-direct-smiles-2p7p-n256-fast` | **Timeout**（10h） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16303778` | `succ-direct-smiles-ood-n256-fast` | 完成（2h49m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16303779` | `succ-direct-smiles-2p7p-rl` | 完成（1h50m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16303783` | `succ-direct-smiles-2p7p-rl-n256` | 完成（3h07m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16338557` | `succ-direct-smiles-2p7p-n256-fast2` | 完成（15h39m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16338559` | `succ-direct-smiles-ood-n256-fast2` | 完成（2h47m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

fast2 2p7p **56.1%**（≈串行 56.2%）；OOD **78.8%**；RL n=256 strict **23.2%**（collapse，不可用）。

## 近期 Slurm Job 一览（2026-06-19~20 direct SMILES preference DPO）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16406939` | `succ-direct-smiles-dpo-2p7p` | 完成（16h08m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

1 epoch preference DPO（pref n=16 + bench n=256）：2p7p strict **52.1%**（vs SFT 56.2%，**-4.1pp**）；7p **18.8%**（vs 25.8%）；无 collapse 但未学到有效 preference（`policy_pref_rate≈50%`）。

## 近期 Slurm Job 一览（2026-06-21 direct SMILES v2 mixed condition）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16472651` | `succ-direct-smiles-2p7p-v2` | **Node Fail**（epoch 9/12，17m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16472652` | `succ-direct-smiles-ood-v2` | 完成（~1h28m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16477041` | `succ-direct-smiles-2p7p-v2`（resume） | 完成（3h17m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

v2（`append_property_program` + property-count curriculum，n=64 推理）：2p7p **68.1%**（+11.9pp vs v1 n=256）；OOD **53.1%**（7p bucket **51%** vs v1 7.0%）。

## 近期 Slurm Job 一览（2026-06-15 hybrid 默认验证）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16072912` | `succ-denovo-2p7p-ours`（standalone hybrid 默认） | 完成（18m49s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `16072913` | `succ-denovo-ood-ours`（standalone hybrid 默认） | 完成（12m42s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |

与 materializer sweep `16056226`/`16056227` 数值一致，确认默认切换可复现。

## 近期 Slurm Job 一览（2026-06-15 random shortlist sanity）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16075242`–`16075253` | `succ-{2p7p,ood}-san-v1_k{64..4096}` | 完成 | [random sanity](understanding-condition/materializer-random-sanity-sweep.md) |

random@k=4096 已达 2p7p **97.8%** / OOD **98.0%**，与 latent hybrid 几乎重合；小 k 下 random 反而更高。

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
