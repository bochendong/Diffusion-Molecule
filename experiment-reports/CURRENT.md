# 当前结论

Date: 2026-08-20

De novo 对照先读 [PAPER-FAIR-PROTOCOL.md](PAPER-FAIR-PROTOCOL.md)。禁止 SketchMol 原文 80.4 / 76.8 / 73.6，禁止 n=2048 ranking 当 `ours`。

编辑论文工作稿：[EDITING-PAPER.md](EDITING-PAPER.md)。

## 现在在做什么

**E1b 已完成并停止。** `20140228` 的关键词与字母乱序对照没有呈现预注册的语言因果模式：keyword real5 72.3%，scrambled 71.5%，所有 language checks 均失败。E1 template/paraphrase 的提升不能再写成语言理解；主表仍是 B41。

**Frontier contribution audit 已完成。** 同一 B39 warm start、同一 B41 架构，只改 next-event 标签。B41 frontier real5 69.5%，canonical-singleton **85.7%**，random-singleton **81.7%**。因此 order-free frontier objective 不是贡献。两个 singleton 数字是在已打开 Table1 上得到的诊断，不能事后挑成主表。

**Fresh confirmation 已预注册。** 下一次只在60个全新 source-disjoint train-only sources 上一次性冻结 B41、canonical、D3-GRPO，以及 template/paraphrase/keyword/scrambled/reversed 语言控制。生成进程不接受 target 路径，exact n=20，无 ranking/oracle selection；结果不用于在同一 sources 上继续调参。

**Particle coverage audit 工程未收尾。** independent/orthogonal/interacting 的 mean unique SMILES 约 8.47/8.48/8.57，差异很小；collector 最后因 mixed JSON key sort 报错。现有证据不能声称 particle interaction 显著增加覆盖。

**D5a 停。** `20099636` 过不了门。不要升级 trust region。主表仍是 B41。localization 冻结。投稿 = B41 方法 + D4 diagnostic。

**D3 停。** 20GB 上 GRPO 跑通了，GSK3B 60.2% 不到 70% 门。主表仍是 B41。SFT 对照已补评（`20085019` 采完后收尾脚本引号崩了）。

## Table1 编辑（n=20 one-shot，Acc@0.65）

真实 5 任务才能对 MolEditRL。合成 HBA 不进主表。

| 任务 | MolEditRL | **Ours B41** | C5 混合（分析） | D0a B31 | D1（stop） | D2（stop） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSK3B↑ | 34.2% | **45.5%** | 100% | 26.3% | 41.4% | 3.1% |
| 官方 40 GSK3B | — | **47.5%** | 100% | 25% | 45% | 2.5% |
| MW↑ | 40.4% | **69%** | 100% | 17% | 52% | 0% |
| SA↓ | 62.8% | **74%** | 64% | 13% | 75% | 10.1% |
| DRD2 | 51.8% | **64%** | 44% | 5% | 65% | 2.4% |
| RB↓ | 63.4% | **94.9%** | 37.4% | 22.2% | 96.0% | 2.7% |
| **真实 5 任务** | **~50.5%** | **69.5%** | 69.1% | 16.7% | 65.9% | 3.7% |
| GSK3B Tanimoto | — | ~0.72 | ~0.86 | ~0.60 | ~0.70 | ~0.99 |

B41：997 条全出、validity 100%、0 skip。五个真实任务全赢 MolEditRL。平均和 C5 打平，但是一条过程。

C5：0.5/0.5 GraphEditDSL + 冻结 B31。GSK3B/MW 100%，RB 37.4% 输给 MolEditRL。不是方法行。

B31-only：GSK3B **any@0.15=100%**，Acc@0.65 只有 26%，Tanimoto ~0.60。C5 的 100% 不是「20 次 B31 抽签」。

D1：MCS 胶水，GSK3B 下降。停。不要加大 energy_weight。

D2：球内短流，61% 复印 source。停。

C9：mixer 线，RB 仍差。不换生成器，不上主表。

## D3（停，分析）

球内监督微调 B41，再 GRPO（oracle 只在训练奖励）。推理仍 n=20、无 MCS、无 ranking。过门 GSK3B≥70% 失败。

| 任务 | B41 | D3 SFT | D3 GRPO |
| --- | ---: | ---: | ---: |
| GSK3B↑ | 45.5% | 52.0% | **60.2%** |
| 官方 40 GSK3B | 47.5% | 57.5% | 57.5% |
| MW↑ | 69% | 67% | 72% |
| SA↓ | 74% | 84% | **88%** |
| DRD2 | 64% | **67%** | 63% |
| RB↓ | 94.9% | 96.0% | **99.0%** |
| **真实 5 任务** | 69.5% | 73.2% | **76.4%** |
| GSK3B Tanimoto | 0.72 | 0.73 | 0.76 |

GRPO 训练奖励从 0.65→0.68，success@0.65 约 25%→27%。GSK3B 涨了，但 60% 到不了 70%。不要再加大 GRPO。不要换主表。

## D4b（门过了，claim 不能写）

冻结 B41，学 `(x,p)→P(EDIT)` hard mask。对照是逐分子 matched-count random，以及保持 edit-subgraph 形状的 shuffled learned。标签来自 MCS 对齐的 source–target，不是最优域。

| 任务 | B41 | learned | random | shuffled | D4a paired-target |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSK3B↑ | 45.5% | 47.5% | 62.6% | **77.8%** | 90% |
| 官方 40 GSK3B | 47.5% | 45% | 60% | **85%** | 92.5% |
| MW↑ | 69% | 72% | **88%** | 85% | 92% |
| SA↓ | **74%** | 70% | 39% | 35% | 66% |
| DRD2 | 64% | 64% | 45% | 39% | 65% |
| RB↓ | 94.9% | **96.0%** | 55.6% | 57.6% | 96.0% |
| **真实 5 任务** | 69.5% | **69.9%** | 58.0% | 58.9% | 82% |
| GSK3B Tanimoto | 0.72 | 0.75 | 0.91 | 0.93 | — |

预注册检查：learned real5 > random、> shuffled，RB≥70%。收集脚本写了 `go_localization_learned`。原因是 random/shuffled 把 RB 砸到 56%，real5 被拖下去；learned 几乎等于冻结 B41。

GSK3B 上位置是反的：同样的 edit 预算，打乱化学位置（shuffled）比学到的位置高 30pp。训练标签 atom EDIT 21%，推理预测 33%。head 学会了「别乱锁」（保 RB），没学会「GSK3B 该锁哪」。不要把 D4b 写成方法，不要换主表。

## D5a（停，不升级）

冻结 B41，非 STOP logit 减去 η。`property_alpha` 从训练对 Tanimoto 学；`constant_eta=2`。validity 100%，0 skip。

| 任务 | B41 | property η | constant η=2 |
| --- | ---: | ---: | ---: |
| GSK3B↑ | 45.5% | 45.5%（η=0，与 B41 同种子） | 46.5% |
| 官方 40 GSK3B | 47.5% | 47.5% | 57.5% |
| MW↑ | 69% | 69% | **78%** |
| SA↓ | 74% | 74% | 71% |
| DRD2 | **64%** | 59%（η=3.0） | 59% |
| RB↓ | 94.9% | 94.9% | 94.9% |
| **真实 5 任务** | **69.5%** | 68.5% | 69.9% |
| GSK3B Tanimoto | 0.72 | 0.72 | 0.75 |

过门失败：GSK3B 没有 +3pp，real5 没有明显高于 B41，property 没有赢 constant。球内 Tanimoto 标准差只有 0.04，p→Tanimoto 不是「GSK3B 该少改」的标签（这次 GSK3B 的 η=0）。软 η 没有砸 RB（和 D4 硬空间 mask 不同），但不够成为第二机制。不要调 scale、不要换成 GNN、不要上 `η(x,p,t)`。

## Latent 线

干净方法就是 B41。互补是分析：fragment 保 GSK3B 性质、图事件保 RB/相似。不能 MCS 加成一个模块。

## De novo

average / raw@1：2p 约 18% vs SketchMol 重跑 75%（**−50pp**）。编辑论文里只当 limitation，不抢 SOTA。
