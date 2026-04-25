# ImagiChem：片段库与骨架（Fragments & Scaffolds）

ImagiChem 在**组装阶段**不是把经验式当成无序字母串随便连成图，而是从预定义的 **骨架（cores）** 和 **可挂接官能团片段（groups）** 出发，在 `MoleculeGraph` 上合并、延伸，再交给 RDKit 整理成合法分子。下面按代码中的两个库说明「都有什么」。

---

## 1. 骨架库 `CORE_LIBRARY`（`ImagiChem code/cores.py`）

- **规模**：共 **1520** 条记录（`choose_core_by_pool` 只在当前原子池能满足 `requirements` 的条目中随机选一条）。
- **结构**：每条包含  
  - `"name"`：标识名  f
  - `"factory"`：返回一个已带坐标与键级的 `MoleculeGraph`  
  - `"requirements"`：从原子池里**必须先扣除**的元素计数（与骨架原子组成一致）

### 1.1 命名与类别（便于在文件里搜索）

| 前缀 | 数量 | 含义（从命名与图结构理解） |
|------|------|---------------------------|
| `bidimensional_ring1` … `bidimensional_ring10` | 10 | 偏「平面」环系：苯并类、含 N/S 的六元/五元杂环等变体 |
| `tridimensional_ring1` … `tridimensional_ring10` | 10 | 带 3D 坐标的稠合/桥环类起始骨架（命名强调三维排布） |
| `core1` … `core1500` | 1500 | 更大、更复杂的稠环 / 多杂原子 / 含卤等**精选骨架**，每条有独立 `coreN_graph` 工厂函数 |

### 1.2 `requirements` 在做什么

组装开始时，若总原子数 ≥ 12，可能选 **1 或 2 个** core（见 `assemble_from_input_string`）。每放入一个 core，就从剩余池子里减去该 core 的 `requirements`，再把图 merge 进总分子。因此骨架不是「任意字符串」，而是**与化学图一一对应的片段**。

### 1.3 如何自己浏览

文件约六万余行：图构建函数在前部，`CORE_LIBRARY` 列表在**文件末尾**。可用编辑器搜索 `"name": "core` 或 `"name": "bidimensional` 跳转。

---

## 2. Functional-group fragment library `GROUP_LIBRARY` (`ImagiChem code/groups.py`)

After the **core(s) and the fallback single-atom seed**, the main loop drains the remaining atom pool: it **prefers attaching whole small fragments** (rather than adding one atom at a time). Entries:

| `name` | Chemistry / topology in the graph | Deducted from pool (`requirements`) | Sampling weight `weight` |
|--------|-------------------------------------|-------------------------------------|--------------------------|
| `ester` | Acyl carbonyl + single-bonded O (ester moiety) | C×1, O×2 | 0.45 |
| `amide` | Amide carbonyl fragment | C×1, O×1, N×1 | 0.45 |
| `amine` | Single N (attachment vertex) | N×1 | 0.25 |
| `alcohol` | Single O | O×1 | 0.20 |
| `nitro` | Nitro group (N + two O) | N×1, O×2 | 0.15 |
| `ether` | O–C sub-graph in code | Listed as N×1, O×2 (**does not match a typical ether stoichiometry; see source as ground truth**) | 0.15 |

Notes:

- Each entry’s `factory` (e.g. `ester_graph`) builds a small graph; `merge_graph_into` attaches it to an atom chosen by `choose_host` on the current molecule.
- `RNG.random() < weight` gates whether a group enters the candidate list, then `RNG.choice` picks among candidates; if no group applies, the code falls back to **single-atom chain growth** (`choose_next_elem_for_chain`).

---

## 3. 和「任意原子串」的差别（设计层面）

| 环节 | 做法 |
|------|------|
| 起始结构 | 从 `CORE_LIBRARY` 合法子集中选 1–2 个**已成键**的骨架图 |
| 中间增长 | 优先用 `GROUP_LIBRARY` 的**成键片段**，否则才单原子拼接 |
| 收尾 | 连通分量合并、`graph_to_rwmol`、`SanitizeMol`、必要时降键级 |

经验式字符串只决定**元素池子**与能否满足各 `requirements`；**真正的连通方式与局部化学环境**由上述库与图操作约束，因此比「纯随机 SMILES 字符」更接近**有化学结构的拼装**。

下游（`run_imagichem_processing`）还会做 **SA score**、**PAINS** 等过滤，进一步偏向可合成、类药讨论中常见的性质空间（具体阈值以 `imagichem_core.py` 为准）。

---

## 4. 相关源码入口

| 文件 | 内容 |
|------|------|
| `ImagiChem code/cores.py` | 全部 core 图工厂 + `CORE_LIBRARY` |
| `ImagiChem code/groups.py` | 官能团小图工厂 + `GROUP_LIBRARY` |
| `ImagiChem code/imagichem_core.py` | `choose_core_by_pool`、`assemble_from_input_string`、`merge_graph_into` |
| `ImagiChem code/graph_utils.py` | `MoleculeGraph`、价键与连接逻辑 |

---

## 5. 本目录其它笔记

- `pixel-line-to-molecular-formula.md`：像素行统计 → 经验式  
- `deterministic-seed.md`：整图哈希 → 随机种子  

若你希望把本页链接进仓库根目录 `README.md`，可在该文件中加一节 **Documentation** 指向 `learn/`。
