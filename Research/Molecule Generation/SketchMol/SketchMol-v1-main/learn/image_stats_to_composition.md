# 图像行统计 → 组成量（C / N / S 与分子规模）：笔记与和本仓库的关系

> **重要**：下面这套「均值 / 标准差 / 对比度 / 峰频 → \(n_C, n_N, n_S, A_{\mathrm{tot}}\)」的**显式公式与门限**，在 **SketchMol-v1 本仓库源码中并未实现**。  
> 本仓库里理化条件来自 **SMILES → RDKit**（见 `data_process/calculate_property.py`），扩散条件为 CSV 中的连续/离散标签，**不是**从分子图按行做 \(\mu_r,\sigma_r\) 反推原子数。

若你在做自己的方法或读另一篇工作，可把本文当作**独立公式备忘**；若要在 SketchMol 里用，需要**自行写预处理脚本**生成条件再接入 `pubchemdata` 一类接口。

---

## 1. 语义对照（你关心的「这部分」）

| 图像量 | 直观角色 | 映射目标（你的设定） |
|--------|----------|----------------------|
| **行均值强度** \(\mu_r\) | 碳骨架笔画/墨量往往占主导 | **碳计数** \(n_C\)（仿射） |
| **行标准差** \(\sigma_r\) | 局部明暗起伏 | **氮丰度** \(n_N\)（仿射） |
| **对比度** \(\mathrm{Contrast}_r\) | 线条与底对比强弱 | **分子规模** \(A_{\mathrm{tot}}\)（近似仿射） |
| **局部峰频** \(f_{\mathrm{peak}}(r)\) | 尖峰/端点密度 | **是否引入硫** \(n_S\in\{0,1\}\)（相对自适应阈值 \(\tau_r\)） |

---

## 2. 数学形式（与你在消息中一致）

设第 \(r\) 行、宽度 \(W\) 上像素强度为 \(I_{rj}\)（\(j=1,\ldots,W\)），可先在每行定义：

**行均值**

\[
\mu_r = \frac{1}{W}\sum_{j=1}^{W} I_{rj}.
\]

**行标准差**

\[
\sigma_r = \sqrt{\frac{1}{W}\sum_{j=1}^{W}\bigl(I_{rj}-\mu_r\bigr)^2}.
\]

**碳计数（仿射模型）**

\[
n_C = a\,\mu_r + b.
\]

**氮丰度（仿射模型）**

\[
n_N = c\,\sigma_r + d.
\]

**分子规模（由对比度近似）**

\[
A_{\mathrm{tot}} \approx \alpha\,\mathrm{Contrast}_r + \beta.
\]

（\(\mathrm{Contrast}_r\) 的具体定义需在你方法里约定：例如 Michelson 对比度、RMS 对比度、或相对背景的峰谷差等。）

**硫：由峰频与自适应阈值决定**

\[
n_S = \mathbf{1}\!\left[f_{\mathrm{peak}}(r) > \tau_r\right],\qquad \tau_r\ \text{为自适应阈值}.
\]

（\(f_{\mathrm{peak}}\) 的实现同样需在方法中定义：例如二阶差分过零、形态学骨架端点密度、或小波峰计数等。）

---

## 3. 和 SketchMol 仓库实际在做什么（避免混读）

| 主题 | 本仓库做法 |
|------|------------|
| 训练标签里的 LogP、QED、分子量、TPSA、氢键等 | `data_process/calculate_property.py`：对 **SMILES** 用 RDKit 计算，写入 CSV |
| 条件扩散 | 读 CSV 连续/离散条件，**与图像行统计无直接对应** |
| 图像上的 `mean` | `ldm/modules/losses/vqperceptual.py` 等处用 `torch.mean(ori_images, dim=1)` 区分**非白区域**（分子大致区域），用于加权重建损失，**不是** \(\mu_r\) 估 \(n_C\) |

因此：你关心的「**行级统计 → 元素组成/规模**」是一套**独立的图像→化学先验或反演假设**；若要实验，需要在数据管线里**先算 \(\mu_r,\sigma_r,\mathrm{Contrast}_r,f_{\mathrm{peak}}\)**，再决定如何与生成模型条件拼接。

---

## 4. 若要在工程上落地（提纲）

1. 读入与训练一致的分子图（与 RDKit 渲染尺寸、反色/白底约定一致）。  
2. 对灰度图按行计算 \(\mu_r,\sigma_r\)；按你定义算 \(\mathrm{Contrast}_r\)、\(f_{\mathrm{peak}}(r)\)。  
3. 用标定集拟合 \((a,b),(c,d),(\alpha,\beta)\)，或直接用物理可解释区间做归一化。  
4. \(\tau_r\) 的自适应策略写清（例如分位数、邻行平滑、或按 \(\sigma_r\) 缩放）。  
5. 将得到的 \(n_C,n_N,n_S,A_{\mathrm{tot}}\)（或连续代理）写入 CSV 新列，再在 config 里接到 `pubchemBase_various_continuousV2` 的条件分支（需改数据类与 yaml）。

---

如需把某一段 **Python 参考实现**（按行 `numpy` 算 \(\mu_r,\sigma_r\) + 一种可选对比度/峰频）也放进 `learn/`，可以说明希望的图像约定（灰度范围、是否反色、行方向是宽还是高）。
