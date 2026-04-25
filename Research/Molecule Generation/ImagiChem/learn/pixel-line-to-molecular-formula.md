# 图像行统计量 → 分子式：实现说明

本文说明 ImagiChem 中如何将单行灰度像素的统计特征映射到经验分子式（碳/氮/硫数量与总原子规模）。对应代码主要在 `ImagiChem code/imagichem_core.py` 的 `PixelLineAnalyzer` 类。

**公式预览**：在 Cursor / VS Code 中请打开「Markdown 预览」并确认设置里 **`Markdown › Math: Enabled`** 为勾选（`markdown.math.enabled`），块级公式使用 `$$`，行内使用单个 `$`。

---

## 论文中的关系式（对照）

$$
n_C = a\,\mu_r + b,\qquad \mu_r = \frac{1}{W}\sum_{j=1}^{W} I_{rj}
$$

$$
n_N = c\,\sigma_r + d,\qquad \sigma_r = \sqrt{\frac{1}{W}\sum_{j=1}^{W}(I_{rj}-\mu_r)^2}
$$

$$
A_{\mathrm{tot}} \approx \alpha\,\mathrm{Contrast}_r + \beta
$$

$$
n_S = \mathbf{1}\!\left[f_{\mathrm{peak}}(r)>\tau_r\right],\qquad \tau_r\ \text{adaptive}
$$

---

## 数据流：像素行从哪里来

`split_pixel_rows` 读取灰度图，按**行**拆成多条一维序列 `pixel_line`，长度 $W$ 即图像宽度（该行像素个数）。

---

## 统计量如何计算

在 `analyze_pixel_pattern` 中：

| Symbol / name | In code | Notes |
|---------------|---------|-------|
| $\mu_r$ | `np.mean(self.pixel_line)` | Mean grayscale along the row |
| $\sigma_r$ | `np.std(self.pixel_line)` | NumPy default `ddof=0`; matches $\sqrt{\frac{1}{W}\sum (I_{rj}-\mu_r)^2}$ |
| $\mathrm{Contrast}_r$ | `np.max(...) - np.min(...)` | Grayscale range (max − min) for that row |
| $f_{\mathrm{peak}}$ (peak count) | `len(_find_peaks())` | Number of local maxima above the relative amplitude threshold |

峰检测 `_find_peaks(threshold=0.3)`：在动态范围 $[\min,\max]$ 上取高度阈值 `min + 0.3 * (max - min)`，仅当像素为严格局部极大且高于该阈值时记为一个峰。行内全为常数时无峰。

---

## 线性映射函数 `_map_value`

所有「$a\mu+b$」形式的离散化都通过同一仿射映射实现：

$$
\text{out} = \frac{v - \text{in\_min}}{\text{in\_max} - \text{in\_min}}(\text{out\_max} - \text{out\_min}) + \text{out\_min}
$$

对应代码中的 `_map_value(value, in_min, in_max, out_min, out_max)`。

---

## 与各原子数、总规模的对应关系

### 碳 $n_C$（均值主导）

- 将 `mean` 从区间 $[0, 255]$ 线性映射到碳数 $[15, 40]$，再 `clip` 到该范围。
- 对应论文中 $n_C = a\mu_r + b$ 的离散化版本（$a,b$ 由端点确定）。

### 氮 $n_N$（标准差映射丰度）

- 将 `std` 从 $[0, 100]$ 线性映射到 N 的允许区间 $[1, 8]$（`atom_ranges` 中的上下界）。
- 对应 $n_N = c\sigma_r + d$ 的离散化。

### 总原子规模 $A_{\mathrm{tot}}$（对比度）

- 将 `contrast` 从 $[0, 255]$ 映射到近似总原子数 `total_atoms_approx`，范围 $[10, 80]$。
- 后续在循环中加入 O、N、S 等时，若总和超过 `total_atoms_approx` 会削减当前原子数。
- 若 `total_atoms_approx - c_count` 过小（`< 5`），会把碳数调整为至少 `max(15, int(total_atoms_approx * 0.5))`，避免其它杂原子没有空间。

### 硫 $n_S$（峰频 + 阈值）

- **峰的定义**：见上文 `_find_peaks`，阈值随该行动态范围变化（可视为一种「自适应」）。
- **是否引入硫**：当前实现为指示函数——若 `num_peaks > 250` 则硫计数为 `1`，否则为 `0`（之后在范围内还会被 `clip`；`atom_ranges` 中 S 为 `(0, 2)`，但此分支通常只产生 0 或 1）。

注意：论文中 $\tau_r$ 若强调「随位置 $r$ 或行宽 $W$ 变化」，与代码里**固定的峰数阈值 250**并不完全同一形式；宽图更容易超过 250，窄行可能很难触发加硫。

### 氧（论文片段未列出，但代码存在）

- 氧原子数同样用 **均值** `mean`，从 $[0, 255]$ 映射到 $[1, 10]$。

---

## 其它实现细节

- `gradient = np.diff(self.pixel_line)` 已计算，但**未参与**分子式生成。
- 卤素在特定行（如 `i % 20 == 0`）由随机数决定，与像素统计无关。
- 得到各原子计数后，将元素符号展开为列表、`shuffle` 再拼接成字符串，供后续 `assemble_from_input_string` 使用。

---

## Quick reference: paper vs. implementation

| Quantity | Paper | This codebase |
|----------|-------|---------------|
| $n_C$ | $a\mu_r+b$ | `mean` → $[0,255]$ → $[15,40]$ |
| $n_N$ | $c\sigma_r+d$ | `std` → $[0,100]$ → $[1,8]$ |
| $A_{\mathrm{tot}}$ | $\alpha\,\mathrm{Contrast}+\beta$ | `contrast` → $[0,255]$ → $[10,80]$ |
| $n_S$ | $\mathbf{1}[f_{\mathrm{peak}}>\tau_r]$ | `num_peaks > 250`; peak-height threshold in `_find_peaks` is relative (row-adaptive) |
| $n_O$ | (not in the equations you quoted) | Same driver as C: `mean` mapped to $[1,10]$ |

### Examples (same `_map_value` linear rule as in code)

Mapping used: $\text{out} = \text{in\_min\_out} + (\text{value}-\text{in\_min}) \cdot \frac{\text{in\_max\_out}-\text{in\_min\_out}}{\text{in\_max}-\text{in\_min}}$, then `int(...)` truncates toward zero.

| Quantity | Sample statistic on one row | Maps to (before other caps) |
|----------|------------------------------|-----------------------------|
| $n_C$ | `mean` $= 0$ | $15$ (dark row → fewer C in range) |
| $n_C$ | `mean` $= 255$ | $40$ |
| $n_C$ | `mean` $= 128$ | $\lfloor 128\cdot\frac{25}{255}+15\rfloor = 27$ |
| $n_N$ | `std` $= 0$ | $1$ |
| $n_N$ | `std` $= 100$ | $8$ |
| $n_N$ | `std` $= 35$ | $\lfloor 35\cdot\frac{7}{100}+1\rfloor = 3$ |
| $A_{\mathrm{tot}}$ | `contrast` $= 0$ | $10$ |
| $A_{\mathrm{tot}}$ | `contrast` $= 255$ | $80$ |
| $A_{\mathrm{tot}}$ | `contrast` $= 200$ | $\lfloor 200\cdot\frac{70}{255}+10\rfloor = 64$ |
| $n_O$ | `mean` $= 128$ | $\lfloor 128\cdot\frac{9}{255}+1\rfloor = 5$ |
| $n_S$ | `num_peaks` $= 12$ | $0$ (no S from this rule) |
| $n_S$ | `num_peaks` $= 300$ | $1$ (then still clipped by `atom_ranges` and total-atom budget) |

The carbon count and heteroatoms are further adjusted if the implied total exceeds `total_atoms_approx` (see sections above); the numbers above are the **direct** linear-map outputs.

---

## 相关文件

- `ImagiChem code/imagichem_core.py`：`split_pixel_rows`、`PixelLineAnalyzer`、`run_imagichem_processing`

---

## 纯文本备用（无数学扩展时）

若环境完全不渲染公式，可用下面等价写法对照：

- $n_C = a \cdot \mu_r + b$，$\mu_r = (1/W) \sum_j I_{rj}$
- $n_N = c \cdot \sigma_r + d$，$\sigma_r$ 为行内灰度标准差（ddof=0）
- $A_{\mathrm{tot}} \approx \alpha \cdot \mathrm{Contrast}_r + \beta$
- $n_S = 1$ 当峰数超过阈值，否则 $0$；峰高阈值按行 min/max 相对取值
