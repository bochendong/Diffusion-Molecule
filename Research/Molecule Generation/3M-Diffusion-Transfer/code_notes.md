# 3M-Diffusion 代码阅读笔记

## 数据入口

主复现实验使用：

```text
Research/Molecule Generation/3M-Diffusion/data/ChEBI-20_data
```

字段格式：

```text
CID    SMILES    description
```

README 里的训练命令都围绕 `ChEBI-20_data/train.txt`、`test_filter.txt` 展开。
仓库还带有 `PubChem324k` 和 `kv_data`，但当前 clone 中它们是较小 split，不是完整
324k 规模。

## Stage-1 graph-text 对齐

相关文件：

```text
Research/Molecule Generation/3M-Diffusion/model/blip2_stage1.py
Research/Molecule Generation/3M-Diffusion/model/blip2qformer.py
```

关键逻辑：

```text
graph_encoder -> graph_proj
text encoder/Q-former -> text_proj
contrastive loss: graph-to-text and text-to-graph
```

对我们的启发：

```text
source molecule image / graph 和 instruction 不应该只是拼接字符串；
应该先训练成可检索、可对齐、可作为 diffusion condition 的 latent/token 表征。
```

## VAE / latent 表征

相关文件：

```text
Research/Molecule Generation/3M-Diffusion/polymers/vae_train.py
Research/Molecule Generation/3M-Diffusion/polymers/poly_hgraph/hgnn_clip.py
```

3M-Diffusion 先用 hierarchical molecule VAE 得到 molecule latent，再让 diffusion
学习这个 latent distribution。我们不一定要复用它的 VAE，但应该复用这个原则：

```text
generation stream 不直接吃离散标签，而是吃经过 Understanding stream 对齐后的 latent condition。
```

## Diffusion 训练

相关文件：

```text
Research/Molecule Generation/3M-Diffusion/polymers/main.py
Research/Molecule Generation/3M-Diffusion/polymers/diffusion/denoising_diffusion.py
Research/Molecule Generation/3M-Diffusion/polymers/poly_hgraph/gine_diffusion.py
```

关键逻辑：

```text
latent, context = bart_model.forward_encoder(...)
loss = diffusion(latent, context, mask)
sample(context) -> latent -> decoder -> SMILES
```

可借鉴点：

```text
condition dropout / null condition
EMA
DDIM sampling
pred_x0 objective
latent normalization
validity / novelty / diversity 评估
```

不建议照搬的点：

```text
它的 context 是 molecule description/text latent；
我们的 context 必须是 source molecule image + edit instruction 的 tokens。
它生成的是 description-conditioned molecule；
我们要生成 source-conditioned edit target。
```

