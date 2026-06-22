# SUCC Benchmark Roadmap and RL + Agentic Plan

| 字段 | 值 |
| --- | --- |
| **状态** | draft for paper planning |
| **最后更新** | 2026-06-22 |
| **范围** | `SketchMol-Understanding-Condition` 主线 benchmark 组织 + 下一阶段方法路线 |

## 先讲结论

我们现在**不是 benchmark 太少**，而是 benchmark **分散、口径不统一**。

仓库里已经有 4 条很有价值的线：

1. **MolEdit-Instruct / Table1 source-conditioned edit**
2. **zero-source de novo 2p-7p**
3. **zero-source OOD**
4. **reverse stimulation / inpainting / SketchMol-style setup**

如果目标是顶会，下一步重点不该只是再多跑几个设置，而是把这些线整理成一套更像论文的层级：

1. **一个必须打穿的主 benchmark**
2. **一个能体现 RL / agentic 价值的 generalization benchmark**
3. **一组体现我们特色的 extension benchmark**

## 当前仓库里已经有的 strongest results

### A. Table1 / MolEditRL 对齐线

来源：[MolEditRL comparison](../moleditrl-baseline-comparison.md)

| 任务组 | 当前最好结果 | 备注 |
| --- | ---: | --- |
| Table1 10-task mean `Acc(0.65)` | **0.894** | SUCC v3 `table_success_rerank` |
| Table1 10-task mean `Acc(0.15)` | **0.854** | 同上 |
| overlap-3 mean `Acc(0.65)` | **0.860** | MolEditRL 对应 0.555 |

这条线的优点是：**可以直接和 MolEditRL 对打**。  
风险也很明确：当前高分很大程度来自 **task-aware rerank**，论文里要非常诚实地区分“生成能力”与“selection gain”。

### B. direct SMILES zero-source de novo

来源：[direct SMILES v2](direct-smiles-denovo-v2-mixed-condition.md)

| Benchmark | 当前最好结果 | 备注 |
| --- | ---: | --- |
| 2p7p overall strict | **0.791** | v2, `n=128`, default decoding |
| OOD overall strict | **0.741** | v2, `n=128`, conservative decoding |
| OOD 7p bucket | **0.720** | 同上 |

这条线的优点是：**真正 zero-source，比较像从文字条件直接做 lead design**。  
当前最大问题不是分数低，而是 **2p7p 最优和 OOD 最优依赖不同 decoding policy**，还不能作为一条“干净主 pipeline”。

### C. latent materializer / hybrid 线

来源：[2p7p benchmark](denovo-2p7p-benchmark.md)、[OOD benchmark](denovo-ood-benchmark.md)

| Benchmark | 当前最好结果 | 备注 |
| --- | ---: | --- |
| 2p7p overall strict | **0.970** | dualmode v1 + `latent_property_rerank` |
| OOD overall strict | **0.985** | dualmode v1 + `latent_property_rerank` |

这条线更像 **diagnostic upper bound**，说明 latent 本身不是唯一瓶颈；  
但它太接近 retrieval/property-oracle，不适合作为最终论文的核心 claim。

## 外部 benchmark landscape

### 1. MolEditRL / MolEdit-Instruct

论文：[MolEditRL](https://arxiv.org/abs/2505.20131) / [HTML](https://arxiv.org/html/2505.20131v1)

它的价值在于：

1. **数据规模大**：MolEdit-Instruct 约 3M editing examples，覆盖 **10 个属性**。
2. **口径清楚**：主表用 `Validity`、`Acc_all` / `Acc_valid`（相似度阈值 `0.65` 和 `0.15`）以及 `FCD`。
3. **既有单属性，也有多属性**，而且已经形成了一个大家能认的 source-conditioned editing 对照面。

对我们来说，这条 benchmark 的意义非常直接：  
**它是最适合做“我们已经超过已有方法”的主表。**

### 2. DrugAssist / MolOpt-Instructions

论文：[DrugAssist](https://arxiv.org/abs/2401.10334)

它更像两件事的 precedent：

1. **instruction-based molecule optimization**
2. **interactive / multi-turn refinement**

它的数据和评测更偏 **single / double-property**，以及对话式 refinement。  
所以它不一定是我们最核心的 SOTA 对打对象，但它非常适合支撑我们后面“**agentic refinement 不是拍脑袋，是有明确先例的**”这个叙事。

### 3. GeLLM3O / MuMOInstruct

论文：[GeLLM3O ACL 2025](https://aclanthology.org/2025.acl-long.1225/) / [PDF](https://aclanthology.org/2025.acl-long.1225.pdf)

这是我觉得**最值得认真对齐**的一条外部 multi-property benchmark 线：

| 维度 | MuMOInstruct |
| --- | --- |
| 总任务数 | **63** |
| 训练中 `>=3` property 的任务 | **42** |
| eval 任务 | **10** |
| eval 划分 | **5 IND + 5 OOD** |
| 重点 | realistic multi-property lead optimization |

它的优点是：

1. 任务更像真实 lead optimization，而不是只测单一属性方向。
2. 已经把 **IND / OOD** 明确拆出来了。
3. 非常适合承接我们现在的 zero-source / multi-property 能力。

如果我们要讲“**generalist molecular optimizer**”，MuMOInstruct/C-MuMOInstruct 这条线很难绕开。

### 4. C-MORAL / C-MuMOInstruct

论文：[C-MORAL](https://arxiv.org/abs/2604.23061) / [HTML](https://arxiv.org/html/2604.23061v1)

这条最重要的不是绝对数值，而是它已经把**RL post-training for molecular optimization**这件事讲顺了：

1. 用 **GRPO / GDPO** 这种 group-relative RL 思路做 post-training。
2. 在 **IND / OOD multi-objective** 设置上报告主结果。
3. 强调 reward 需要处理 **heterogeneous property scales** 和 **competing objectives**。

这基本就是我们后面做 RL 的最强参照系。  
换句话说，如果我们真要做 `RL + agentic`，最自然的落点不是“再做一版 REINFORCE”，而是：

1. **multi-sample group-relative RL**
2. **reward normalization across properties**
3. **把 revise / verify loop 变成 agentic outer loop**

## 现在这些 benchmark，哪些该放论文主表

### Tier 1: 必须打的主表

#### 1. MolEdit-Instruct Table1

原因：这是当前我们最能直接对标 MolEditRL 的地方。

论文里建议主打：

1. **10-task mean**
2. 单任务表
3. 结构保持 / 相似度 / FCD 之类的配套指标

但要明确分成两层：

1. **base generator**
2. **with selection / rerank**

否则 reviewer 很容易追着问：你到底是在比生成，还是在比一个 task-aware search pipeline。

#### 2. MuMOInstruct 或 C-MuMOInstruct 风格的 5 IND + 5 OOD

原因：这能把我们从“会做几个特定 benchmark”拉到“**对复杂 multi-property 任务有系统 generalization 能力**”。

如果短期内还接不进官方数据，最差也应该先在 repo 内部搭一个**同结构代理 benchmark**：

1. IND: 用当前可见 property combo
2. OOD: hold out 组合 / extreme specs / rare combos
3. 统一 one-shot pipeline

#### 3. unseen-property / held-out-property adaptation

MolEditRL appendix 风格的 unseen task 是值得补的。  
因为这可以回答一个 reviewer 一定会问的问题：

> 模型到底是在背 seen combinations，还是学到了 transferable molecular editing knowledge？

### Tier 2: 体现 agentic 价值的主实验

这部分才是我们区别于纯 SFT / 纯 RL baseline 的地方。

建议统一成一个**固定 budget**的 protocol：

| 设置 | 含义 |
| --- | --- |
| 1-turn | 一次生成，不修正 |
| 2-turn | 生成 -> verifier feedback -> revise |
| 3-turn | 再来一轮 revise |
| fixed oracle budget | 每个 prompt 允许的 property evaluation 次数固定 |

主指标建议至少报：

1. success / strict
2. validity
3. first-shot success
4. revision gain
5. average oracle calls

这样 agentic 的贡献才会从“口头上说有 feedback loop”，变成**可量化的 sample-efficiency / revision-efficiency 提升**。

### Tier 3: 作为 extension 的特色实验

这部分更适合做亮点补充，而不是扛主结论：

1. reverse stimulation
2. inpainting
3. SketchMol-style sketch-to-molecule
4. video-like / progressive condition reveal

这些实验很有特色，但如果没有 Tier 1/Tier 2 托底，单独拿去打顶会会显得 benchmark ecosystem 不够硬。

## 我对方法主线的建议

### Stage A: 先收敛出一条干净的 one-shot pipeline

这一步不是最花哨，但必须先做。

目标：

1. **一个 checkpoint**
2. **一个 decoding policy**
3. **一个 candidate selection rule**
4. 同时覆盖 edit、2p7p、OOD

如果这个基础没有统一，后面所有 RL / agentic 的增益都会显得不稳。

### Stage B: RL 不要直接做 token-level REINFORCE，改成 group-relative

我们之前 direct SMILES v1 的 RL collapse 已经说明，  
**裸 REINFORCE + sparse strict reward** 很容易把模型带歪。

更合理的下一版应该是：

1. 每个 prompt 采样一个 candidate group
2. verifier 评估每个 candidate 的 property satisfaction / validity / diversity
3. 在组内做 relative advantage，而不是直接用绝对 reward

reward 建议分 task 类型：

### edit task reward

1. property satisfaction
2. source similarity / scaffold preservation
3. validity
4. novelty 只给很小权重

### zero-source de novo reward

1. property satisfaction
2. validity
3. uniqueness / diversity
4. optional synthesizability regularizer

这样 edit 和 de novo 不会被同一套 reward 生硬绑死。

### Stage C: agentic 不要做“聊天味很重”的 agent，要做 verifier-driven molecule loop

我更建议把 agentic 定义得工程化一点：

1. **Planner**：把自然语言条件解析成 property program / hard-soft constraints
2. **Generator**：产出一批候选
3. **Verifier**：算 property gap、invalid 原因、冲突属性
4. **Reviser**：根据 verifier 输出，改下一轮 prompt / constraint emphasis / decoding mode

也就是说，我们不是做一个“会说话的化学 agent”，而是做一个：

**LLM-centered iterative molecule optimization loop**

这会更像真正的科研方法，而不是包装。

## 具体建议的论文主故事

一句话版：

> We build an LLM-centered molecular optimization system that first learns a strong one-shot generator, then improves controllability with group-relative RL, and finally boosts hard-task success through verifier-guided agentic revision.

对应三层贡献：

1. **strong one-shot base**
2. **RL alignment for multi-objective controllability**
3. **agentic revision for hard or OOD prompts**

## 接下来最值得做的 4 步

### Step 1. 固定 paper benchmark hierarchy

先定死这三层，不再反复摇摆：

1. MolEdit-Instruct Table1
2. MuMO/C-MuMO 风格 multi-property IND/OOD
3. repo-native extension（2p7p / reverse stimulation / inpainting）

### Step 2. 先把 one-shot main pipeline 统一

这里的标准不是“单 benchmark 最优”，而是：

1. 同一 pipeline 在 edit / 2p7p / OOD 都不掉穿
2. 可以作为后续 RL / agentic 的 base model

### Step 3. 做 group-relative RL v1

建议第一版只做：

1. zero-source 2p7p + OOD
2. direct SMILES backbone
3. group-based relative reward
4. 保留 SFT anchor / KL

不要一上来 edit、de novo、OOD 全混在一起训，不然很难定位。

### Step 4. 在 hard subset 上做 agentic revise benchmark

最先值得做 agentic 的，不是 easiest rows，而是：

1. 6p / 7p
2. rare_combo
3. reverse stimulation
4. unseen combination

因为这些地方最能体现 iterative correction 的价值。

## 当前不建议作为主 claim 的东西

1. **hybrid materializer 98% 这条线**  
   太像 retrieval/property-oracle amplification，更适合作为 diagnostic upper bound。

2. **benchmark-dependent decoding policy**  
   2p7p 用 default、OOD 用 conservative 可以当 ablation，但不适合做主 pipeline。

3. **简单 REINFORCE RL**  
   目前已有 collapse 证据，继续堆这条线收益不高。

## 一个更现实的近线目标

如果按“1-2 个月内形成可投稿故事”来排优先级，我会这么定：

1. **保住 MolEdit-Instruct / Table1 优势**
2. **补齐一个更标准的 multi-property IND/OOD benchmark**
3. **做出不 collapse 的 RL v1**
4. **在 hard subset 上做出 agentic revise 的明确增益**

做到这四件事，论文故事就会从“我们在 repo 里做了很多实验”变成：

> 我们提出了一条从 one-shot generation 到 RL alignment 再到 agentic refinement 的完整 molecular optimization pipeline，并在 editing、multi-property generalization、OOD 和 iterative correction 上都给出可复现实证。

## 参考链接

1. MolEditRL: [arXiv](https://arxiv.org/abs/2505.20131), [HTML](https://arxiv.org/html/2505.20131v1)
2. DrugAssist: [arXiv](https://arxiv.org/abs/2401.10334)
3. GeLLM3O / MuMOInstruct: [ACL Anthology](https://aclanthology.org/2025.acl-long.1225/), [PDF](https://aclanthology.org/2025.acl-long.1225.pdf)
4. C-MORAL: [arXiv](https://arxiv.org/abs/2604.23061), [HTML](https://arxiv.org/html/2604.23061v1)
