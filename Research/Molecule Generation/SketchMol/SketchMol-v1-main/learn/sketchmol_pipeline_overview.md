# SketchMol 流程概览（简版）

这份文档用一个「从分子到图像，再回到分子」的视角，快速说明 SketchMol 的核心流程。

---

## 1) 目标与核心思路

SketchMol 想解决的问题是：  
在分子设计里，同时建模**局部化学细节**（原子、键）和**全局结构布局**（整体骨架），并支持可控生成与编辑。

核心思路：

- 把分子当作可视对象（2D 图像）来建模；
- 在图像空间中完成生成/编辑；
- 再把结果图像还原成可用的分子表示（如 SMILES）；
- 最后做有效性与性质筛选。

---

## 2) 训练阶段（离线）

### Step A: 数据准备

```python
records = load_pubchem()
for mol in records:
    mol_std = canonicalize(mol)                 # Standardize structure (desalt/normalize)
    img = render_2d(mol_std, size=256)          # Use a unified coordinate system and canvas size
    img = center_crop_resize_norm(img)          # Crop -> resize -> normalize to [-1, 1]

    cond = {
        "activity": get_activity_label(mol_std),
        "props": calc_props(mol_std)            # QED / SA / logP, etc.
    }

    save_train_sample(image=img, condition=cond, target_smiles=to_smiles(mol_std))
```


### Step B: 视觉生成模型学习

关键点：SketchMol 不是直接在像素空间的原图 `x0` 上做 diffusion，  
而是先用第一阶段自编码器把图像编码成 latent `z0`，再在 latent 空间执行加噪与去噪。

```python
model = ConditionalUNet()
optimizer = AdamW(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    # x0: molecule image, z0: encoded latent, cond: property/activity condition
    for x0, cond in train_loader:
        z0 = encode_to_latent(x0)                        # image -> latent
        t = sample_timestep(batch_size=len(x0))          # random diffusion step
        eps = randn_like(z0)                             # Gaussian noise in latent space
        zt = q_sample(z0, t, eps)                        # forward diffusion to z_t

        eps_pred = apply_model(zt, t, cond)              # inject condition and predict noise

        loss_simple = mse(eps_pred, eps)                 # main diffusion objective
        loss = l_simple_weight * loss_simple

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

训练时可以这样理解：

- 每个 batch 都会随机采样扩散步 `t`，让模型看到不同噪声强度下的 latent 表示；
- 主任务是“预测加进去的噪声”，学会后就能在采样时反向去噪；
- 条件控制主要通过 `apply_model(zt, t, cond)` 注入网络，在采样阶段再配合 guidance scale 增强；
- 定期从纯噪声采样图像，并用结构解析模块回转成分子，检查有效性与条件命中率。

### Step C: 图像到结构的还原能力

- 使用分子图识别模块（如 MolScribe 路径）把图像解析回结构表达；
- 确保后续可以把生成结果转成可计算、可筛选的分子对象。


```python
for t = T → 1:
    ε̂ = model(z_t, t, cond)
    z_{t-1} = reverse_step(z_t, ε̂)
```
---

## 3) 推理阶段（在线生成）

### 路径 1：从零生成（de novo）

1. 输入条件（可选）：例如目标性质、任务标签；
2. 从随机噪声开始扩散采样，得到分子图像；
3. 图像解析回 SMILES/分子图；
4. 做合法性检查（能否解析、价态是否合理等）；
5. 计算性质并筛选候选分子。

### 路径 2：局部编辑（inpainting）

1. 给定已有分子图像；
2. 对希望修改的区域打掩码；
3. 模型只重绘掩码区域，保留其余结构；
4. 解析并验证新分子；
5. 比较编辑前后性质变化，做定向优化。

---

## 4) 后处理与评估

常见评估会关注：

- **有效性（Validity）**：生成结果是否是化学上可用的分子；
- **唯一性（Uniqueness）**：候选是否多样而非重复；
- **新颖性（Novelty）**：是否跳出训练集已有样本；
- **目标性质达成度**：如活性、QED、SA 等指标是否满足约束。

---

## 5) 一句话总结

SketchMol 的流程可以理解为：

`分子结构 -> 2D 图像表示 -> 扩散生成/编辑 -> 图像解析回分子 -> 化学筛选与优化`

它的价值在于把「生成」和「编辑」统一到同一视觉空间里，让分子设计更直观，也更容易做局部可控修改。
