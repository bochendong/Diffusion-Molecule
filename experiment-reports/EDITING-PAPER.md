# Source-conditioned graph-jump editing

工作稿。投稿英文；本文件是锁死的 claim 和表。先读 [PAPER-FAIR-PROTOCOL.md](PAPER-FAIR-PROTOCOL.md)。

| 字段 | 值 |
| --- | --- |
| **论文类型** | 分子**编辑**方法文，不是 de novo 生成器 |
| **主表 `ours`** | **B41**：一条图事件跳过程，valid-terminal STOP，一次物化 |
| **不是 `ours`** | C5 家族混合、C9、D1 MCS 胶水、D2 短流、n=2048 ranking、UniVideo retrieval |
| **协议** | MolEdit Table1，n=20，Acc@0.65 any@20，无 ranking、无 oracle 挑选、无二次编辑 |
| **主表只报真实 5 任务** | GSK3B↑、MW↑、SA↓、DRD2（三性质）、RB↓。合成 HBA 不进主表 |

仓库实现名 B41 不要写进标题。对外方法名：

**Viability-preserving graph-jump editing**：源分子进冻结图 AE，条件化事件核在价键/芳香支持上跳，粒子间互斥保持覆盖，合法 STOP 时解码一次 SMILES。

---

## 0. 禁止写进投稿

- SketchMol 原文 2p 80.4 / 3p 76.8 / 4p 73.6 当对照。
- 用 C5 的 GSK3B 100% 冒充本方法。那是 GraphEditDSL + 冻结 B31 的 0.5/0.5 混合。
- 用 n=128 / n=2048 / finalizer / ranking 当一次生成。
- 宣称 de novo 打赢 SketchMol。公平 raw@1 / average 差约 **−50pp**（[PAPER-FAIR-PROTOCOL.md](PAPER-FAIR-PROTOCOL.md) 表 A）。
- 把 D1 写成「把能量接到跳过程上」。D1 是 MCS 投影，GSK3B Acc@0.65 从 45.5% 降到 41.4%。
- 把 D2 写成方法。短流塌成复印 source。

---

## 1. Title / abstract（可直接改英文）

**Title.** Source-Conditioned Molecular Editing with a Viability-Preserving Graph Jump Process

**Abstract (draft).**
Property-conditioned molecular editing must change a source molecule while remaining similar to it. We edit in a frozen graph-latent autoencoder with a single jump process: a condition-dependent event kernel proposes atom and bond events on chemically viable support, and a learned STOP emits one molecule when the graph is materializable. On the MolEdit Table1 protocol (20 raw attempts, Acc@0.65, no oracle selection), the method beats MolEditRL on all five real tasks and matches the average of a two-family discrete mixer while using one process. Fragment-energy editing reaches perfect GSK3B success only by leaving the Tanimoto 0.65 ball; inference-time MCS fusion of the two mechanisms lowers GSK3B. We do not contest image-space de novo generation.

---

## 2. Contributions（三条，不要第四条「统一生成」）

1. A single source-conditioned graph jump process with viability support and valid-terminal STOP; one decode per attempt; no ranking, repair, or property oracle in the loop.
2. On MolEdit Table1 n=20 Acc@0.65, **all five real tasks exceed MolEditRL**, with 100% validity and source Tanimoto ≈ 0.72 on GSK3B.
3. A mechanism split: fragment energy vs graph events. The mixer (C5) spikes GSK3B to 100% by mixing two generators. Gluing them with MCS (D1) does not compose. C5 is analysis, not the proposed method.

---

## 3. 方法（写成一条过程，不要 B22–B41 流水账）

推理合同（与评测一致）：

1. 源 SMILES → 图 AE 编码（冻结）。条件来自性质指令（source-only，不含 target 分子）。
2. 初始化 n=20 条粒子（与 any@20 的 20 次 attempt 一一对应，不是从更大池里再挑）。
3. 每步：事件核在当前图的合法支持上采样事件（改原子/键、有限出生配额）；价键与芳香规则裁掉不可物化的 STOP。
4. 直接采样 20 条固定预算粒子。粒子互斥是实现细节；最新 coverage audit 只有约 8.47→8.57 unique 的小变化，不能单独列为贡献。
5. 学到的 STOP 且图可物化 → 解码一次 SMILES。禁止 retry、repair、oracle、ranking。

训练：在编辑对上拟合事件前缀，STOP 边际推动「先做完该做的事件再停」。表上数字来自冻结 checkpoint 的 Table1 评测（`outputs/d0_b41_table1_n20/`），不是训练时看过 Table1 测试。

实现对应仓库：`viability_preserving_interacting_particle_transport` + Table1 评测时的 valid-terminal STOP。投稿里用上述过程描述，附录可写 checkpoint 名。

---

## 4. 主表（锁死）

同一 `table1_test_rows.csv`，n=20，Acc@0.65 any@20。MolEditRL 为论文官方行。真实 5 任务均值是五个 Acc 的算术平均。

| 任务 | MolEditRL | **Ours (graph jump)** | Mixer (C5, analysis) | Fragment energy (B31) |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 34.2 | **45.5** | 100 | 26.3 |
| 官方 40 GSK3B | — | **47.5** | 100 | 25 |
| MW↑ | 40.4 | **69.0** | 100 | 17.0 |
| SA↓ | 62.8 | **74.0** | 64.0 | 13.0 |
| DRD2 | 51.8 | **64.0** | 44.0 | 5.0 |
| RB↓ | 63.4 | **94.9** | 37.4 | 22.2 |
| **真实 5 任务** | **50.5** | **69.5** | 69.1 | 16.7 |
| GSK3B Tanimoto | — | ~0.72 | ~0.86 | ~0.60 |
| Validity | — | 100 | 100 | 100 |

数字来源：

- Ours：`outputs/d0_b41_table1_n20/summary.json`（GSK3B n=99、RB n=99；官方 GSK3B n=40 为 47.5%）
- Mixer：`outputs/joint_graph_fragment_categorical_c5/summary.json`（家族约 49.8% fragment / 50.2% graph）
- Fragment：`outputs/d0_b31_only_table1_n20/summary.json`（any@0.15 GSK3B = 100%，Acc@0.65 = 26.3%）

**怎么讲主表：** 对 MolEditRL 五个任务全赢。平均与 mixer 打平（69.5 vs 69.1）。Mixer 的 GSK3B 100% 和 RB 37.4% 不是同一生成器；fragment 单独 GSK3B Acc@0.65 只有 26.3%，因为切出了 0.65 球（Tanimoto ~0.60，any@0.15 却是 100%）。

合成 HBA 任务可放附录，不进主表。

---

## 5. 分析表（不要当方法）

| 实验 | 作用 | 结论 |
| --- | --- | --- |
| B31-only | fragment 能量单独 | GSK3B 性质对了但相似度不够 |
| C5 | 两家族 0.5/0.5 + Graph GRPO | 平均高，RB 输给 MolEditRL，不是一条过程 |
| D1 | MCS 把 B31 目标糊进 B41 事件网格 | 对齐率 70.9%；GSK3B 45.5→41.4；MW 69→52。不组成 |
| D2 | 球内短 rectified flow | 61% 复印 source。停 |
| Frontier audit | B41 ready-set vs singleton next-event label | canonical 85.7%、random 81.7%、frontier 69.5%；frontier objective 不是贡献，singleton 只能在 fresh confirmation 后决定是否升级 |
| E1/E1b | 冻结语言 projector 与关键词/乱序控制 | 对照不分离，不能声称 LLM/语言理解 |

D1 细节：`outputs/d1_b31_energy_on_b41_table1_n20/summary.json`，real5 65.9%。不要加大 energy_weight 再报一版。

---

## 6. De novo（limitation，一段）

不是本方法的主结果。公平口径：[PAPER-FAIR-PROTOCOL.md](PAPER-FAIR-PROTOCOL.md)。

SketchMol 重跑 raw@1 / average：2p 约 74–75%。我们 Direct 同口径约 17–18%。差约 −50pp。any@20 是预算合同，他们 2p 已饱和到 99.7%，我们 84.7%，仍输 15pp。图像扩散把离散分子变成连续像素再 OCR，是更强的 **de novo 生成器**。本工作做 **编辑**。

---

## 7. Related work 怎么划界

- **MolEditRL**：同一 Table1 协议，主对照。
- **SketchMol**：de novo 图像扩散，相关工作 + limitation，不当编辑 SOTA 对照。
- GraphEditDSL / fragment energy：分析里的两种机制，不是 proposed method。
- 不要把 UniVideo / 检索重排写成 baseline 胜利。

---

## 8. 还缺什么（写稿时补，不是再发明生成器）

必须补：

1. 过程图：source → 事件跳 → STOP → SMILES（一列，20 粒子并行）。
2. 任务剖面图：五任务条形图，MolEditRL / Ours / Mixer / Fragment。
3. 方法正文去掉 B22–B41 编号，改成「事件核、化学支持、valid-terminal STOP、固定 n=20 直接采样」四个模块。
4. 一次 source-disjoint fresh confirmation，先冻结 B41/canonical/D3 与完整语言对照，再打开 target；未通过前不得把 canonical 85.7% 或 E1 template 提升写成主贡献。

有则更好，没有也能投：

- 再加 1 个公开编辑 baseline（同一 n=20 合同）。没有就只用 MolEditRL + 机制消融。
- 少量定性编辑例子（RB 剪侧链 vs GSK3B 仍贴源）。

不要补：

- 新家族混合、新 MCS、新短流、图像 OCR 编辑器、ranking。

---

## 9. Intro 论点（三段）

1. 编辑不是生成：必须留在源的相似球里（Tanimoto 0.65 是本协议的成功门槛）。
2. 现有离散编辑器要么切太远（fragment 能量），要么靠两个生成器混合才能两头都高。
3. 我们用一条带化学支持的图跳过程，在同一预算下五个真实任务都超过 MolEditRL。

---

## 10. 数字核对

| 行 | Acc@0.65 | 文件 |
| --- | --- | --- |
| B41 GSK3B | 0.454545… | d0_b41 summary |
| B41 real5 | 0.694808… | 同上 |
| C5 real5 | 0.690747… | c5 summary |
| B31 GSK3B | 0.262626… | d0_b31 summary |
| D1 GSK3B | 0.414141… | d1 summary |
