# SketchMol Understanding-Condition Stream

当前 UniVideo-style image-to-structure 完整实验的运行说明见：
[`README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md`](README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md)。

## 0. 核心判断

这个方向可以做，但不能把故事讲成“旁边加一个 LLM”。比较稳的研究切入点是：

> 用 MLLM 把分子图像 + 自然语言编辑目标解析成一组可被 diffusion cross-attention 使用的 condition tokens，从而提升分子编辑里的 scaffold 保留、局部替换、性质优化和目标蛋白活性控制。

换句话说，大模型不负责直接生成 SMILES，也不只是输出一段 caption；它负责生成 conditional latent / condition token sequence。diffusion generator 仍然负责生成分子图像，MolScribe 或后处理再把图像还原成结构。

这和 UniVideo 的思路最像：

- UniVideo: image/video/text -> Qwen-VL hidden states/metaqueries -> video diffusion transformer。
- SketchMol: molecule image/sketch/text/property -> molecular understanding tokens -> latent diffusion UNet。

SketchMol 现有条件流已经是 cross-attention：`MixedEmbedderV2` 把离散/连续性质变成条件 token，配置里的 `context_dim` 是 256。因此最自然的改法不是推翻 SketchMol，而是把 MLLM condition tokens 接到同一个 cross-attention 接口里。

## 0.1 当前主线：不用小模型，改用大 VLM

当前主实验不再以小 CNN / hashed text proxy 作为方法主体。那些结果只保留为
pipeline sanity check 和 ablation 负控。

## 0.2 独立 unified 3M-Diffusion 版本

3M-Diffusion 启发的 unified 训练已独立到 `SketchMol-Unified-3MDiffusion/`。
**当前默认 benchmark / 训练数据是 MolEdit-Instruct enhanced splits**（见
`datasets/README.md` 与 `submit_unified_moledit_pipeline.sh`）。旧的
multi-property manifest 仍用于 image / VLM pipeline，但不再是 Unified 3M 的主线默认。

## 0.3 UniVideo-style 双流生成模型

根据 UniVideo 的训练方式，现在新增了更贴近最终主线的双流模型：

```text
Understanding stream:
  source molecule image / source SMILES / instruction
    -> frozen HF VLM condition features
    -> connector
    -> diffusion-readable condition tokens C'

Generation stream:
  source molecule image / SMILES
    -> molecule-image VAE
    -> source latent z_src, shape 4x32x32
    -> source-conditioned latent diffusion
    -> generated target latent z_tgt
    -> VAE decoder
    -> generated target molecule image
```

这里的 VAE backend 对齐 SketchMol 的 latent 接口：输入 256x256 分子图像，
latent 是 `4 x 32 x 32`。如果 edit dataset 没有现成 PNG，会用 RDKit 从
SMILES 在内存里渲染分子图，不额外制造海量小文件。

代码入口：

```text
sketchmol_understanding_condition/molecule_image_vae.py
sketchmol_understanding_condition/univideo_molecule.py
scripts/train_molecule_image_vae.py
scripts/train_univideo_molecule_generation.py
scripts/run_univideo_molecule_pipeline.sh
scripts/submit_univideo_molecule_pipeline.sh
```

训练分三段，对应 UniVideo 的 staged training 思路：

```text
Stage 1: connector alignment
  只训练 connector，让 frozen VLM hidden/query features 对齐 target latent、
  target properties、property deltas、active mask、edit direction 和 similarity bin。

Stage 2: diffusion fine-tuning
  训练 connector + source-conditioned diffusion denoiser，用 target latent 的
  pred_noise diffusion loss 学生成。

Stage 3: multi-task/dropout
  加 condition dropout 和 source dropout，模拟 classifier-free guidance/多任务
  条件缺失，避免模型只记某一种输入形式。
```

一键提交与完整运行说明见
[`README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md`](README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md)。
OCR 跑不通时，同一文档 §6 有 OCR-free materialized benchmark。

当前 canonical 输出目录：

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/
  univideo_molecule/eval_latent/
  univideo_molecule/image_structure_benchmark/
  univideo_molecule/benchmark_materialized_primary_fast/   # materialized benchmark
```

最小提交示例：

```bash
SUCC_HF_MODEL_NAME_OR_PATH=/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct \
SUCC_CONDITION_ROWS=SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv \
bash SketchMol-Understanding-Condition/scripts/submit_univideo_molecule_pipeline.sh
```

## 代码入口与 legacy 实验

完整脚本列表、HF VLM pipeline、mixed-objective 旧实验和 encoder v0–v2.2
历史结果已从本 README 移除，避免和当前 MolEdit / materialized benchmark 主线
混在一起。需要时直接看：

```text
README_UNIVIDEO_IMAGE_STRUCTURE_EXPERIMENT.md   # UniVideo 运行手册
SketchMol-MultiProperty-EditDataset/README.md   # image 数据集构建
datasets/README.md                              # MolEdit-Instruct 数据根
SketchMol-Unified-3MDiffusion/README.md         # Unified 3M + MolEdit 训练
```

保留的核心脚本：

```text
scripts/run_univideo_molecule_pipeline.sh
scripts/submit_univideo_molecule_pipeline.sh
scripts/run_univideo_materialized_benchmark.sh
scripts/submit_univideo_materialized_benchmark.sh
scripts/submit_hf_vlm_multiproperty_pipeline.sh
scripts/evaluate_univideo_image_benchmark.py
scripts/materialize_univideo_target_molecules.py
```

## 1. 从 UniVideo 迁移什么

UniVideo 值得迁移的不是视频模型本身，而是三个设计原则。

### 1.1 MLLM 不是输出文本，而是输出 hidden condition

UniVideo 的 `mllm_encoder.py` 用 Qwen2.5-VL，配置里有 `num_metaqueries`。tokenizer 会在 prompt 后追加 `<begin_of_img><img0>...<imgN><end_of_img>` 这类 metaquery tokens，然后取 MLLM hidden states 作为 generation stream 的条件。

这对 SketchMol 的启发是：不要让 LLM 输出“把 logP 降低一点”这种文本后再人工解析；而是让模型学习一组固定长度的 molecular edit query tokens，例如 16 或 32 个 query token，投影到 SketchMol UNet 的 `context_dim=256`。

### 1.2 Understanding stream 必须看输入图像

UniVideo 配置里区分 `mllm_use_ref_img` 和 `mllm_use_cond_pixels`，说明 generation stream 的条件可以同时来自文本指令和参考图像/像素内容。

对应到 SketchMol：

- 输入：原始分子图像或手绘 sketch。
- 输入：自然语言编辑目标，例如“保留 core scaffold，将 para 位氯替换为甲氧基，同时降低 LogP”。
- 输出：一组 condition tokens，供 diffusion inpainting / editing 使用。

这里的关键是 MLLM 必须参与理解原始分子图像里的结构约束，而不是只读文本目标。

### 1.3 生成流只消费 condition，不关心语言细节

UniVideo 的 transformer forward 接受 `encoder_hidden_states` 和 `encoder_attention_mask`，先用 projection/refiner 处理，再进入 cross-attention。

SketchMol 也类似：当前 `MixedEmbedderV2` 输出 `[B, condition_len, 256]`，UNet 通过 `conditioning_key: crossattn` 消费。我们只需要让新的 MLLM condition encoder 输出同样形状，就能最小侵入接入。

## 2. 推荐研究问题

不要一开始做“任意自然语言分子设计”。风险太大，数据也不一定够。

更好的问题是：

> 给定分子图像和编辑指令，生成满足指令的分子变体，同时尽量保留 scaffold，并改善一个或多个目标性质。

任务可以拆成三类，逐级推进：

1. Scaffold-preserving functional group replacement  
   例如保留 core scaffold，只替换 side chain 或特定位点的 functional group。

2. Property-directed molecular editing  
   例如降低 LogP、提高 QED、降低 TPSA、提高 solubility proxy。

3. Protein-aware optimization  
   例如增强 AKT1/EP4/ROCK1 binding，同时控制 LogP/QED/SA。

其中第 1 类最适合作为 proof of understanding，因为可以很明确地评估“是否真的理解了 scaffold 和编辑位点”。

## 3. 最低可行系统

### 3.1 模型结构

建议先做一个三段式结构：

```text
input molecule image + edit instruction
        |
        v
Molecular MLLM Encoder
        |
        v
condition tokens: [B, Nq, 256]
        |
        v
SketchMol latent diffusion UNet
        |
        v
edited molecule image -> MolScribe/RDKit validation
```

具体模块：

- `MolecularMLLMConditionEncoder`
  - backbone 可以先用冻结的 Qwen2.5-VL 或更轻的 vision-language encoder；
  - 增加 learnable molecular query tokens；
  - 输出 query hidden states；
  - 用 MLP 投影到 `context_dim=256`。

- `HybridConditionEncoder`
  - 保留 SketchMol 原来的 property tokens；
  - 拼接 MLLM tokens；
  - 输出 `[property_tokens + mllm_tokens]` 给 UNet cross-attention。

- `SketchMol diffusion generator`
  - 第一阶段不改 VAE；
  - 第一阶段不改 UNet 主体；
  - 只改 condition stage，降低工程风险。

### 3.2 两种接入方式

最稳的是从轻到重做。

#### A. Adapter-only

冻结 SketchMol diffusion 和 MLLM，只训练一个 projector / adapter，把 MLLM hidden states 映射到 256 维 condition tokens。

优点：

- 训练成本低；
- 可以快速验证 MLLM 信息是否可用；
- 适合数据较少的情况。

缺点：

- 能力上限有限；
- 如果 SketchMol 原模型没见过这类条件 token，效果可能不稳定。

#### B. Condition-stage finetune

冻结 VAE，微调 UNet cross-attention 和新的 condition encoder。

优点：

- 更容易让 generator 学会使用新 token；
- 适合论文主实验。

缺点：

- 需要更多数据；
- 必须做严格消融，否则容易被质疑只是多参数带来的收益。

## 4. 数据怎么构造

这个方向成败主要取决于数据，不取决于模型名字。

### 4.1 自监督编辑对

从已有分子库构造 `(source molecule, target molecule, edit instruction)`：

1. 用 Bemis-Murcko scaffold 或 MCS 找 source/target 的共同 scaffold。
2. 找 side chain / functional group 差异。
3. 自动生成 instruction：
   - “keep the scaffold and replace chloro with methoxy”
   - “preserve the core ring system and reduce LogP”
   - “modify the side chain to improve QED while keeping molecular weight similar”
4. source 渲染成图像，target 渲染成监督图像。
5. target 计算性质，作为评估标签或辅助 condition。

这样不需要人工标注大规模文本，但 instruction 和结构变化之间有真实对应关系。

### 4.2 性质驱动 pair mining

对同 scaffold 分子按性质差异配对：

- source 和 target scaffold 相同或 MCS 高；
- target 的 LogP 更低 / QED 更高 / TPSA 更合理；
- structural distance 不要太大，避免变成 de novo generation。

这类数据适合证明“自然语言性质目标 + 图像理解”能指导局部优化。

### 4.3 蛋白目标数据

SketchMol 已有 EP4/AKT1/ROCK1 这类 protein condition 使用方式。可以把 protein-aware task 放在第二阶段：

- instruction 写成 “improve AKT1 binding while preserving the scaffold”；
- condition 里同时给 protein token / predicted activity label；
- 评估用 docking score、QSAR predictor 或已有活性标签。

不要把蛋白 binding 作为第一阶段主卖点。它的噪声更大，证明链条更长。

## 5. 怎么证明 understanding stream 真的有用

这是论文最关键的部分。必须设计消融，让评审无法说“LLM 只是装饰”。

### 5.1 必做 baseline

1. SketchMol original condition  
   只用原始 property/protein condition，不用 MLLM。

2. Text-only LLM condition  
   MLLM 只读 instruction，不看 molecule image。

3. Image-only condition  
   MLLM 只看 molecule image，不读 instruction。

4. Random/frozen query tokens  
   用同样数量的 query tokens，但不来自 MLLM。

5. Caption bottleneck  
   让 MLLM 先生成文本 caption，再用文本 encoder 作为 condition。这个 baseline 用来证明 hidden/metaquery 比“LLM 写一句话”更强。

6. Oracle property condition  
   直接给 target property 数值，检验 MLLM 是否只是学到了性质标签。

如果 full model 只比 1 好，但不比 2/4/6 好，这个故事就不成立。

### 5.2 必做反事实实验

1. Instruction swap  
   同一个 source image，交换不同编辑指令。输出应随指令改变。

2. Image swap  
   同一个 instruction，换 source image。输出应保留各自 scaffold，而不是生成同一种结果。

3. Scaffold mask test  
   把 scaffold 区域遮住或打乱，性能应明显下降。否则模型可能没有真的看结构。

4. Functional group target test  
   指定替换基团，评估生成结果里目标 group 出现率和非目标区域保留率。

5. Attention / token attribution  
   检查 diffusion cross-attention 是否使用 MLLM query tokens，尤其是在被编辑区域附近。

### 5.3 指标

基础分子指标：

- validity
- uniqueness
- novelty
- reconstruction / parse success
- property success rate

编辑任务指标：

- scaffold preservation rate
- MCS similarity
- Tanimoto similarity range
- edit locality score
- target functional group hit rate
- non-target region preservation

性质优化指标：

- delta LogP
- delta QED
- delta TPSA
- synthetic accessibility
- multi-objective success rate

蛋白任务指标：

- predicted activity improvement
- docking score improvement
- activity-success under scaffold constraint

最重要的不是单个指标最高，而是 full model 在“需要图像理解 + 指令理解同时成立”的指标上显著优于 baseline。

## 6. 训练目标

第一阶段可以不做复杂 RL 或 docking-in-the-loop。

建议训练目标：

1. Diffusion denoising loss  
   标准 latent diffusion 噪声预测。

2. Condition dropout / classifier-free guidance  
   随机 drop instruction、image、property condition，用于训练可控 guidance 和消融。

3. Optional contrastive alignment  
   让 source + correct instruction 的 condition tokens 比 source + wrong instruction 更接近 target edit embedding。

4. Optional scaffold consistency loss  
   用 RDKit/MolScribe 后验评估做离线筛选，不建议一开始端到端反传。

第一版重点应放在数据和消融，不要把训练目标堆太多。

## 7. 推荐实验路线

### Phase 1: 不改大模型，证明接口可行

- 冻结 SketchMol；
- 冻结 MLLM；
- 训练 MLLM hidden -> 256 condition projector；
- 数据只做 scaffold-preserving replacement；
- 对比 original condition、text-only、random-query。

成功标准：

- validity 不明显下降；
- scaffold preservation 高于 de novo/property-only；
- functional group hit rate 高于 text-only/random-query。

### Phase 2: 微调 condition + cross-attention

- 解冻 SketchMol UNet 里的 cross-attention；
- 加 condition dropout；
- 加 instruction/image swap evaluation；
- 扩展到 property-directed editing。

成功标准：

- 同 scaffold 性质优化成功率提升；
- image swap / instruction swap 都有明显响应；
- target 和 non-target 区域的权衡优于原 SketchMol。

### Phase 3: 蛋白目标故事

- 加 AKT1/EP4/ROCK1 数据；
- instruction 里加入 binding target；
- 与原 single-protein condition baseline 比较；
- 用 docking/QSAR 做二级筛选。

成功标准：

- 在 scaffold preservation 约束下，activity/docking 指标改善；
- 不是简单牺牲 drug-likeness 换 binding score。

## 8. 最大风险和对应处理

### 风险 1: MLLM 不懂分子图

普通 VLM 对分子图像的化学语义可能很弱。处理办法：

- 先用 MolScribe / RDKit 解析 source image，给 MLLM 辅助文本：SMILES、atom labels、scaffold summary；
- 或者预训练一个轻量 molecular image encoder；
- 不要完全依赖通用 VLM 的化学知识。

### 风险 2: 数据规模不够

处理办法：

- 用自监督 pair mining 自动造数据；
- 从简单编辑开始；
- 冻结大模型，只训 adapter；
- 把“能否理解指令和图像”作为主贡献，而不是追求 SOTA de novo generation。

### 风险 3: LLM 变装饰

处理办法：

- baseline 必须强；
- 做 image/text swap；
- 做 random query；
- 做 caption bottleneck；
- 做 attention/token attribution；
- 评估必须包含编辑局部性和 scaffold 保留，而不只是 property 变好。

### 风险 4: 生成结果不可解析

处理办法：

- 保留 SketchMol 原来的图像生成和解析路径；
- 先做 inpainting/editing，不做完全开放生成；
- 后处理用 MolScribe + RDKit validity；
- 对无效样本单独报告，不要只筛选后报成功率。

## 9. 最适合写成论文的故事

标题方向可以是：

> Understanding-Conditioned Molecular Diffusion for Instruction-Guided Scaffold-Preserving Editing

故事线：

1. 现有分子 diffusion 可以按性质生成，但对自然语言编辑目标和图像中具体结构关系理解不足。
2. UniVideo 类 unified understanding-generation 架构启发我们把 MLLM hidden states 作为 generation condition。
3. 我们提出 molecular understanding stream，将 molecule image + instruction 编码成 conditional query tokens。
4. diffusion stream 在 latent image space 生成编辑后分子。
5. 通过严格消融证明：只有同时使用图像理解和指令理解，才能在 scaffold 保留、functional group 命中和性质优化上稳定提升。

这比“LLM + diffusion 生成分子”更容易成立，因为贡献点清楚：MLLM 负责理解和条件构造，diffusion 负责生成，评估专门证明理解流的必要性。

## 10. 近期可执行 TODO

1. 用 `submit_univideo_materialized_benchmark.sh` 在 canonical v2 输出上跑 OCR-free benchmark。
2. 用 `submit_unified_moledit_pipeline.sh` 在 MolEdit-Instruct enhanced splits 上训练 Unified 3M。
3. 用 `evaluate_moledit_table_metrics.py` 对齐 MolEditRL 表格指标。
4. 保留 image pipeline 作为并行对照，但不再把旧 encoder v0–v2.2 实验日志写回本 README。
