# 论文公平口径（必读）

| 字段 | 值 |
| --- | --- |
| **状态** | 已有数字，不再重采 |
| **最后更新** | 2026-08-18 |
| **用途** | 禁止再用 SketchMol 原文 80.4 / 76.8 / 73.6 当 `ours` 对照。主表必须用本文件的重跑行。 |

打开实验报告、写 de novo 数字、和 SketchMol 比较之前，先读这张表。

## 协议

同一 6000 条 2p–7p 条件、同一脚本 `evaluate_denovo_2p7p_budget_sweep.py`。

| 方法 | 候选池 |
| --- | --- |
| **SketchMol 重跑** | `SketchMolBenchmark/outputs/sketchmol_denovo_2p7p_at40_v1/budget_sweep/`（240,000 OCR 候选） |
| **我们 Direct group-RL** | `SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_zero_safe_n40_v1/benchmark_direct_smiles_group_rl/budget_sweep/`（240,000 SMILES 候选） |
| **UMTP 同一 decoder** | `unified_molecular_transformation_policy_v1/eval/train_seed_7/eval_seed_101/2p7p/denovo_2p7p/n20/`（正式 6000 条，n=20） |

两种指标不能混：

- **average / raw@1**：每个生成分子都算。对 SketchMol 生成的公平比。n 从 1 到 40 几乎不动。
- **any@K / best-of-K**：K 个里至少一个 strict。和 MolEditRL Table1 同一推理合同。SketchMol 单次已 ~75%，any@20 会饱和到 ~99%。

仓库里 **没有** 高 n=1。正式 n=1 / raw@1 的 2p 最高约 **20%**。97% / 96.5% / 88.5% 分别是 n=128、n=128 rerank、n=20 finalizer。

原文表格 2p **80.4%** / 3p **76.8%** / 4p **73.6%** 不是上述任一指标的精确复现，**不能当对照行**。

## 表 A：Average / 单分子（对 SketchMol 的公平比）

| 口径 | 2p SketchMol | 2p 我们 | 2p 差 | 3p SketchMol | 3p 我们 | 3p 差 | 4p SketchMol | 4p 我们 | 4p 差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **raw@1** | 74.1 | 16.6 | **−57.5pp** | 67.8 | 12.6 | **−55.2pp** | 56.5 | 7.4 | **−49.1pp** |
| **average@40** | 75.1 | 18.1 | **−56.9pp** | 64.8 | 12.5 | **−52.2pp** | 57.2 | 9.4 | **−47.7pp** |
| 原文（勿对照） | 80.4 | — | — | 76.8 | — | — | 73.6 | — | — |

UMTP n=20 **raw**（不挑）：2p **19.8%** / 3p **15.1%** / 4p **11.7%**，与 Direct average 同量级。

**1→40 的 average 变动**：SketchMol 2p 74.1→75.1（+1.0pp）；我们 16.6→18.1（+1.5pp）。按 average 比，n=1 和 n=20 是同一场，差大约 **−50pp**。

## 表 B：Any@K（和 Table1 同一预算）

### 2p

| K | SketchMol | 我们 Direct | 差 | 我们相对 n=1 | 他们相对 n=1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 74.1 | 16.6 | **−57.5pp** | 0 | 0 |
| 2 | 88.1 | 29.0 | −59.1pp | +12.4pp | +14.0pp |
| 4 | 96.0 | 48.7 | −47.3pp | +32.1pp | +21.9pp |
| 8 | 99.1 | 65.7 | −33.4pp | +49.1pp | +25.0pp |
| **20** | **99.7** | **84.7** | **−15.0pp** | **+68.1pp** | +25.6pp |
| 40 | 100.0 | 92.7 | −7.3pp | +76.1pp | +25.9pp |

### 3p

| K | SketchMol | 我们 Direct | 差 |
| ---: | ---: | ---: | ---: |
| 1 | 67.8 | 12.6 | **−55.2pp** |
| 4 | 93.2 | 35.7 | −57.5pp |
| 8 | 97.0 | 54.4 | −42.6pp |
| **20** | **99.2** | **74.8** | **−24.4pp** |
| 40 | 99.7 | 86.6 | −13.1pp |

### 4p

| K | SketchMol | 我们 Direct | 差 |
| ---: | ---: | ---: | ---: |
| 1 | 56.5 | 7.4 | **−49.1pp** |
| 4 | 87.5 | 28.3 | −59.2pp |
| 8 | 94.6 | 43.8 | −50.8pp |
| **20** | **98.6** | **65.9** | **−32.7pp** |
| 40 | 99.4 | 78.9 | −20.5pp |

我们 2p 从 17% 拉到 85%（**+68pp**）是选出来的；他们从 74% 到 99.7% 是饱和。n=20 仍输。

## 表 C：UMTP 同一 decoder，正式 n=20

| 口径 | 2p | 3p | 4p |
| --- | ---: | ---: | ---: |
| UMTP raw（不挑） | 19.8 | 15.1 | 11.7 |
| **UMTP any@20 / finalizer** | **88.5** | **81.3** | **73.6** |
| SketchMol 重跑 any@20 | 99.7 | 99.2 | 98.6 |
| SketchMol 重跑 average | 75.1 | 64.8 | 57.2 |
| Direct any@20 | 84.7 | 74.8 | 65.9 |

## 表 D：论文怎么报

| 主表 | 我们报什么 | 对照谁 | 合法？ |
| --- | --- | --- | --- |
| **Table1 编辑** | any@20 Acc@0.65 | MolEditRL 官方 any@20 | **合法。** 对方协议就是这个。 |
| **de novo 2p–4p（方法合同）** | any@20，与编辑同一预算 | SketchMol **重跑** any@20（99.7 / 99.2 / 98.6） | 合同统一。他们饱和。我们 2p 仍输约 15pp。 |
| **de novo 2p–4p（对生成器）** | raw@1 和 average | SketchMol **重跑** raw@1 / average（2p 74 / 75） | **这才是公平生成对比。** 差约 −50pp。不要引用原文 80.4。 |

any@20 是「这个 unified 方法怎么用」。raw@1 / average 是「一次生成谁更准」。两列都要。
