# MolScribe：像素坐标 → 离散词表（化学字母表 + 位置槽）

在 SketchMol 自带的 **MolScribe**（`evaluate/molscribe/`）里，模型不是直接吃「原始像素统计」，而是把**图上的原子位置**先变成 **\[0, 1\] 归一化坐标**，再**量化成有限个整数 ID**，与 **SMILES 字符**（化学字母表）拼在同一条离散序列里训练/解码。下面把这部分「从几何到 token」的核心逻辑单独摘出，便于对照源码。

---

## 1. 概念：两层映射

| 层级 | 含义 | 在本项目中的落点 |
|------|------|------------------|
| 图像 → 几何 | 从结构图得到每个原子的 \((x, y)\)，通常已按图像高宽归一化到 \([0,1]\) | Indigo 渲染图 + 标注管线；训练时 `graph['coords']` |
| 几何 → 词表 ID | 把连续坐标量化成 `coord_bins` 个格子，得到整数；字符用 `vocab_*.json` 里的 ID | `evaluate/molscribe/tokenizer.py` 中 `NodeTokenizer` / `CharTokenizer` |

**化学字母表**：`stoi`（string → int）、`itos`（int → string）。`chartok_coords` 默认加载 `vocab/vocab_chars.json`：含 `<pad>`、`<sos>`、`<eos>`、`<unk>`、`<mask>` 以及 SMILES 可能出现的单字符（键号、括号、元素字符等）。

**位置槽**：在 `len(stoi)` 之后的 ID 区间里，为 **x**、**y**（或合并策略）各预留 `input_size`（即 `args.coord_bins`，与 `args.input_size` 一致）个离散 bin，对应 `x_to_id` / `y_to_id`。

---

## 2. 词表布局与 `offset`

`NodeTokenizer` 里，**所有化学/特殊符号占前 `offset = len(self.stoi)` 个 ID**；坐标 token 从 `offset` 开始顺延。

```python
# 逻辑摘要（见 tokenizer.py 中 NodeTokenizer）
@property
def offset(self):
    return len(self.stoi)

def __len__(self):
    if self.sep_xy:
        return self.offset + self.maxx + self.maxy
    else:
        return self.offset + max(self.maxx, self.maxy)
```

- **`sep_xy=False`（常见）**：x、y 共用同一套 bin 长度 `max(self.maxx, self.maxy)`，通过 `is_x` / `is_y` 的区间规则区分（实现里 `is_y` 在 non-sep 时与 x 共用 `offset` 段，由解码顺序约束）。
- **`sep_xy=True`**：x 占 `[offset, offset+maxx)`，y 占 `[offset+maxx, offset+maxx+maxy)`，互不重叠。

`get_tokenizer` 里 `chartok_coords` 使用 `CharTokenizer` + `vocab_chars.json`：

```507:524:evaluate/molscribe/tokenizer.py
def get_tokenizer(args):
    tokenizer = {}
    for format_ in args.formats:
        if format_ == 'atomtok':
            if args.vocab_file is None:
                args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab/vocab_uspto.json')
            tokenizer['atomtok'] = Tokenizer(args.vocab_file)
        elif format_ == "atomtok_coords":
            if args.vocab_file is None:
                args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab/vocab_uspto.json')
            tokenizer["atomtok_coords"] = NodeTokenizer(args.coord_bins, args.vocab_file, args.sep_xy,
                                                        continuous_coords=args.continuous_coords)
        elif format_ == "chartok_coords":
            if args.vocab_file is None:
                args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab/vocab_chars.json')
            tokenizer["chartok_coords"] = CharTokenizer(args.coord_bins, args.vocab_file, args.sep_xy,
                                                        continuous_coords=args.continuous_coords)
    return tokenizer
```

---

## 3. 核心：归一化坐标 ↔ 整数 ID（「像素统计」的离散化）

假设 \(x, y \in [0, 1]\) 已相对图像尺寸归一化（与 `input_size` 网络输入一致的量纲约定）。量化公式：

```python
def x_to_id(self, x):
    return self.offset + round(x * (self.maxx - 1))

def y_to_id(self, y):
    if self.sep_xy:
        return self.offset + self.maxx + round(y * (self.maxy - 1))
    return self.offset + round(y * (self.maxy - 1))

def id_to_x(self, id):
    return (id - self.offset) / (self.maxx - 1)

def id_to_y(self, id):
    if self.sep_xy:
        return (id - self.offset - self.maxx) / (self.maxy - 1)
    return (id - self.offset) / (self.maxy - 1)
```

**含义**：把 \([0,1]\) 上的位置统计成 **`coord_bins` 档**的类别 ID，与 NLP 里的「字符 one-hot 前的一步」同类，只是字母表后半段是「网格坐标」而不是字母。

对应源码位置：`evaluate/molscribe/tokenizer.py` 中 `NodeTokenizer` 的 `x_to_id` / `y_to_id` / `id_to_x` / `id_to_y`（约第 164–178 行）。

---

## 4. 与化学序列的拼接方式

### 4.1 原子级 token + 坐标（`atomtok_coords`，`NodeTokenizer`）

每个原子符号一个 ID，其后跟 \(x\_id, y\_id\)（训练时由 `coords` 填入）。

```python
# nodes_to_sequence： (x,y) + 整颗原子的 symbol → [x_id, y_id, symbol_id]
def nodes_to_sequence(self, nodes):
    coords, symbols = nodes['coords'], nodes['symbols']
    labels = [SOS_ID]
    for (x, y), symbol in zip(coords, symbols):
        assert 0 <= x <= 1 and 0 <= y <= 1
        labels.append(self.x_to_id(x))
        labels.append(self.y_to_id(y))
        labels.append(self.symbol_to_id(symbol))
    labels.append(EOS_ID)
    return labels
```

### 4.2 字符级 + 坐标（`chartok_coords`，`CharTokenizer`）

SMILES 按**字符**展开；仅在「原子 token」后插入两个坐标 ID（见 `smiles_to_sequence` 里对 `is_atom_token` 的分支）。

```python
# 逻辑摘要：每个字符进 stoi；每个 atom token 后追加 x_to_id, y_to_id
def smiles_to_sequence(self, smiles, coords=None, mask_ratio=0, atom_only=False):
    tokens = atomwise_tokenizer(smiles)
    labels = [SOS_ID]
    atom_idx = -1
    for token in tokens:
        ...
        for c in token:
            labels.append(self.stoi[c] if c in self.stoi else UNK_ID)
        if self.is_atom_token(token):
            atom_idx += 1
            if coords is not None and atom_idx < len(coords):
                x, y = coords[atom_idx]
                labels.append(self.x_to_id(x))
                labels.append(self.y_to_id(y))
    labels.append(EOS_ID)
    return labels, indices
```

完整实现见 `CharTokenizer.smiles_to_sequence`（`tokenizer.py` 约 418–451 行）。

### 4.3 栅格视角：`nodes_to_grid`

若把离散格点当作像素索引，也可把「符号 ID」写在 \((i,j)\) 格点上（用于某些网格监督或可视化思路）：

```209:216:evaluate/molscribe/tokenizer.py
    def nodes_to_grid(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        grid = np.zeros((self.maxx, self.maxy), dtype=int)
        for [x, y], symbol in zip(coords, symbols):
            x = round(x * (self.maxx - 1))
            y = round(y * (self.maxy - 1))
            grid[x][y] = self.symbol_to_id(symbol)
        return grid
```

这里 **`round(x * (maxx-1))` 与 `x_to_id` 中的量化一致**，把归一化坐标落到与 `coord_bins` 对齐的整数格点。

---

## 5. 解码约束：`get_output_mask`

在离散坐标模式下，自回归解码每一步只允许合法 token 类型（先 x 再 y 再符号等），通过 `get_output_mask` 把「化学字母表段」与「坐标段」分开约束。见 `NodeTokenizer.get_output_mask` / `CharTokenizer.get_output_mask`（`tokenizer.py` 约 180–190、372–381 行）。

---

## 6. 与「图像像素预处理」的关系（避免混淆）

- **CNN 输入支路**：图像经 `get_transforms`（灰度、Resize、`input_size`、Normalize）得到张量，这是**连续像素特征**。
- **本条文档**：描述的是 **监督信号 / 解码目标**——把 **2D 位置**和 **SMILES 字符**压进**同一个离散词表**，供 Transformer 自回归预测。

两者通过 **`input_size`（或 `coord_bins`）与坐标归一化约定**对齐：坐标必须与渲染图、resize 后的画布一致地归一化到 \([0,1]\)，量化后才与训练标签一致。

---

## 7. 源文件一览

| 文件 | 作用 |
|------|------|
| `evaluate/molscribe/tokenizer.py` | `Tokenizer` / `NodeTokenizer` / `CharTokenizer`，坐标量化与序列编解码 |
| `evaluate/molscribe/vocab/vocab_chars.json` | `chartok_coords` 字符词表 |
| `evaluate/molscribe/vocab/vocab_uspto.json` | `atomtok` / `atomtok_coords` 子结构级词表 |
| `evaluate/molscribe/dataset.py` | `_process_chartok_coords` 等，把标注接到 tokenizer |

---

**说明**：若你本地 `tokenizer.py` 里 `assert self.stoi[...]` 被破坏成无法解析的片段，应恢复为对 `PAD` / `SOS` / `EOS` / `UNK` 等键的正常下标访问（与文件顶部常量一致），否则 Python 无法运行。
