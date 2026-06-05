# 3M-Diffusion 对我们项目的可迁移点

这个文件夹用于整理 3M-Diffusion 对当前 SketchMol Understanding + Generation
方向的启发。结论先说在前面：3M-Diffusion 不适合作为我们的直接 benchmark，
但它非常适合作为“如何把分子语言/结构表征接入 latent diffusion”的参考。

## 1. 数据集有什么可以借鉴

3M-Diffusion 的核心数据形态是：

```text
CID + SMILES + molecule description
```

它不是只给数值性质，而是把分子和自然语言描述配对起来。这一点对我们很重要：
如果我们的数据只保留 2-7 个 property values，Understanding stream 学到的更像
表格条件编码；如果加入可读的 edit instruction、source molecule context、target
property deltas、结构相似度约束，它才更像一个可被大模型理解和泛化的分子编辑任务。

对我们最值得借鉴的是四点：

1. **每条样本保留结构和语言的双视角。**
   我们的数据不应该只是 `source_smiles, target_smiles, property columns`，还应该
   固定保存 `source image/source smiles + instruction + active properties + target
   values + target smiles`。这会让 Understanding stream 有明确的语言监督。

2. **把自然语言条件作为训练对象，而不是只作为展示文案。**
   3M-Diffusion 用 description 对齐分子结构。我们可以把 instruction 设计成更强的
   edit condition，例如：

   ```text
   Given the source molecule image, increase QED and reduce LogP while keeping
   the generated molecule structurally similar to the source.
   ```

   中文 report 可以说：我们不是简单做 property-conditioned generation，而是把
   source molecule、性质编辑目标和结构保留目标统一成一条可理解的条件输入。

3. **保留大规模 text-structure pretraining 数据作为辅助训练。**
   3M 的 ChEBI/PubChem 风格数据可以用来预训练 Understanding stream 的
   molecule-language alignment，但不能直接替代我们的 edit dataset。更合理的做法是：

   ```text
   Stage A: 用 ChEBI/PubChem 描述数据训练 molecule/text 对齐能力
   Stage B: 用我们的 source-target edit pairs 训练 edit-aware condition tokens
   Stage C: 用 edit-aware condition tokens 条件化 diffusion generation
   ```

4. **继续避免海量小图片文件。**
   3M 的数据主要是 TSV 文本文件，不依赖每条样本一个 PNG。这个和我们现在的 inode-safe
   方向一致：主数据集应以 CSV/JSONL/Parquet 保存，图片只在 VLM batch 内从 SMILES
   临时渲染。

## 2. 模型架构有什么可以借鉴

3M-Diffusion 的训练链路可以概括成：

```text
graph/text contrastive encoder
    -> hierarchical molecule VAE latent
    -> text-conditioned latent diffusion
    -> molecular decoder
```

对我们来说，不能直接照搬 text-to-molecule，因为我们的任务是：

```text
source molecule image + edit instruction
    -> condition tokens
    -> generation diffusion stream
    -> target molecule
```

可以借鉴的是它的“分阶段 + latent diffusion”思想：

1. **先训练/固定 Understanding stream，再训练 Generation stream。**
   3M 不是从原始文本直接训练 diffusion，而是先得到 graph-text aligned latent。
   我们也应该先让 Understanding stream 学会：

   ```text
   source molecule image
   instruction
   active property mask
   property direction/delta
   source-target similarity target
   ```

   然后把这些压成一组 condition tokens，交给 diffusion。

2. **condition 应该是 token sequence，而不是单个 pooled vector。**
   3M 使用 Q-former/graph-text 表征做对齐。我们之前的 frozen VLM pooled feature
   可以作为 baseline，但主方法更应该输出 query tokens：

   ```text
   [source structure token] [edit objective tokens] [property delta tokens]
   [similarity-preservation token]
   ```

   这比一个 pooled embedding 更适合 cross-attention。

3. **使用 classifier-free conditioning dropout。**
   3M 的 diffusion 训练里有 unconditional/null condition 的机制。我们可以借鉴为：

   ```text
   训练时随机丢掉 instruction 或 source image condition
   采样时用 guidance weight 控制条件强度
   ```

   这样后续可以做 ablation：无 source、无 instruction、完整条件、不同 guidance。

4. **latent diffusion 比直接生成 SMILES 更符合我们的主线。**
   我们的核心不能变：Understanding stream + diffusion generation stream。
   3M 支持这个方向，因为它本质上也是在 latent space 做条件扩散，而不是让 LLM
   直接吐 SMILES。

## 3. 训练方式有什么可以借鉴

建议改成三阶段训练，而不是只训练一个检索/重排头：

```text
Stage 1: Understanding alignment pretraining
  输入: molecule image/SMILES/rendered graph + description/instruction
  目标: image/text/graph 表征对齐，能识别分子结构和编辑语义

Stage 2: Edit-aware condition token training
  输入: source image + instruction
  监督: target properties, property deltas, active-property mask,
        source-target Tanimoto bin, target latent/retrieval target

Stage 3: Conditioned diffusion generation
  输入: source image + instruction -> condition tokens
  生成: target molecule latent/image/structure
  评估: SketchMol strict success + source Tanimoto constrained success
```

这里最关键的变化是：Understanding stream 不只是 benchmark 里的 reranker，而是
真正成为 diffusion 的条件生成器。

## 4. 对我们当前方向的具体建议

我建议下一步不要把 3M-Diffusion 当成新 baseline 去硬跑，而是借它改造我们的
大实验设计：

1. **数据层：新增 unified condition dataset。**
   把两类数据统一成一个 schema：

   ```text
   type=description_pretrain:
     molecule_image/smiles + description -> molecule identity

   type=edit_generation:
     source_image/source_smiles + instruction + property targets -> target_smiles
   ```

   这样既有 3M 风格的 language-molecule pretraining，也有我们自己的 edit task。

2. **模型层：从 pooled VLM feature 升级到 query-token connector。**
   当前 Qwen2.5-VL pooled/query tokens 可以继续保留，但主方法应该显式训练一个
   edit connector，把 VLM hidden states 变成 diffusion cross-attention tokens。

3. **训练层：加入 target latent / target fingerprint 的对齐损失。**
   不要只预测 property delta。可以同时预测：

   ```text
   target Morgan fingerprint
   target property vector
   source-target Tanimoto bin
   active-property mask
   edit direction
   ```

   这会迫使 Understanding stream 学结构编辑，而不是只学性质分类。

4. **评估层：保持和 SketchMol 可比较。**
   主表仍然是 2p-7p strict success，但必须同时报告：

   ```text
   mean source Tanimoto
   strict@Tanimoto>=0.4/0.6/0.8
   validity / uniqueness / novelty
   ```

   其中 validity/novelty/diversity 可以借 3M-Diffusion 的评估习惯，strict success
   和 source Tanimoto 是我们相对 SketchMol 的核心增量。

## 5. 最终判断

3M-Diffusion 给我们的最大启发不是“多加一个文本描述数据集”，而是：

> 先把分子结构和语言条件对齐到一个 latent space，再让 diffusion 在这个 latent
> space 里做条件生成。

这正好支持我们的主线：大 VLM/Understanding stream 负责读懂 source molecule image
和 instruction，generation diffusion stream 负责生成 target molecule。区别是，
3M 做的是 description-to-molecule，我们要做的是 source-conditioned,
multi-property, similarity-constrained molecule editing。

