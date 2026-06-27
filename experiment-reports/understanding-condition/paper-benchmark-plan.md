# Paper Benchmark Plan

| 字段 | 值 |
| --- | --- |
| **状态** | active planning |
| **最后更新** | 2026-06-26 |
| **项目** | `SketchMol-Understanding-Condition` |
| **目标** | 把当前结果整理成可投顶会的 benchmark 结构，并提供可直接提交到服务器的命令 |

## 设计原则

主文必须明确区分两类能力：

1. **one-shot / direct generation**
2. **assisted generation**（rerank / retrieval / oracle-like shortlist）

不要把 `table_success_rerank`、`latent_property_rerank` 这类 assisted 结果和纯 one-shot 结果混在同一张主表里。

## 主文 Benchmark 结构

| 表格 | 任务 | 主要对比 | 我们的方法行 |
| --- | --- | --- | --- |
| Table 1 | MolEdit Table1 source-conditioned edit | MolEditRL, BioT5, DrugAssist, MolGen, REINVENT4, GeLLM3O | `ours-direct`, `ours-assisted` |
| Table 2 | MuMO / C-MuMO multi-property IND/OOD | GeLLM3O, GeLLMO-C, DrugAssist, MolGen, REINVENT4 | `ours-one-shot`, `ours-group-rl` |
| Table 3 | direct de novo 2p-7p | SketchMol ref + repo baselines | `SFT n128`, `group RL n128`, `group RL n256` |
| Table 4 | direct OOD | repo baselines + future external OOD rows | `default`, `conservative`, `group RL`, `group RL + conservative` |

## P0: 立刻要跑的实验

### 1. Table1 公平版重跑

目的：给当前最强的 v3 MolEdit 线补一条**不依赖 table-success rerank**的公平版结果。

结果位置：

- pack: `.../dataset/table1_benchmark_fair/`
- eval latent: `.../univideo_molecule/table1_eval_latent_fair/`
- benchmark: `.../univideo_molecule/benchmark_materialized_table1_fair/`
- metrics: `.../univideo_molecule/moledit_table_metrics_table1_fair/`

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_v3_fair_table1_extension.sh
```

成功标准：

- 10-task coverage 保持
- 给出 `Acc_all(0.65)` / `Acc_all(0.15)` 的公平版
- 能和 `table_success_rerank` 结果并排，量化 selection gain

### 2. OOD group RL + conservative decoding（n=128）

目的：检验 group RL checkpoint 在 conservative decoding 下，是否能同时保持 OOD overall 提升并修复 7p bucket。

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_benchmark.sh
```

成功标准：

- overall strict > `0.756`
- 7p bucket > `0.590`
- 最理想是同时超过 conservative SFT 的 `0.741 / 0.720`

### 3. OOD group RL n=256（default decoding）

目的：测试 OOD 线的 sample-budget ceiling。

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_n256_benchmark.sh
```

成功标准：

- overall strict > `0.756`
- validity 不显著掉
- 判断 OOD 的瓶颈更像 training 还是 decoding/sample budget

### 4. OOD group RL n=256 + conservative decoding

目的：测试当前最有机会成为 OOD 主结果的组合。

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_n256_benchmark.sh
```

成功标准：

- 同时刷新 OOD overall strict 和 7p bucket
- 如果这条线最好，则作为主文 OOD 主结果

## P1: 下一批要接的外部 benchmark

### 5. MuMO / C-MuMO benchmark port

目的：把我们的方法放到外部论文已经使用的 IND/OOD multi-property benchmark 上。

当前状态：

1. `adapter v1` 已实现，记录见 [external-multiproperty-benchmark.md](external-multiproperty-benchmark.md)。
2. 已覆盖 GeLLMO/MuMOInstruct 和 GeLLMO-C/C-MuMOInstruct 的 5 IND + 5 OOD task specs。
3. 已新增 source-conditioned CSV exporter、external generated-property evaluator、server submit 入口。
4. 默认 pilot 关闭 output-side property rerank，避免把外部 benchmark 第一版做成 assisted result。

预期产出：

- `outputs/mumo_port_v1/`
- `outputs/cmumo_port_v1/`

提交命令：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/mumo_or_cmumo_test.json \
SUCC_EXTERNAL_MULTIPROP_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

注意：BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等性质需要 external generated-property CSV 才能公平评估；没有 oracle CSV 时只作为 coverage / plumbing pilot。

### 6. PMO small-budget pilot

目的：为后续 RL / agentic 提供 sample-efficiency benchmark。

先做 5 个代表 task 的小子集：

1. 低预算：100 / 500 / 1000 oracle calls
2. 高预算：10000 oracle calls
3. 比较 `one-shot`, `group RL`, `agentic 2-step`, `agentic 4-step`

## P2: 特色 extension（适合 supplementary）

1. reverse stimulation
2. inpainting
3. SketchMol-style sketch-to-molecule

这些更适合作为补充亮点，不建议先于 P0 / P1。

## 当前 one-shot 主线建议

### Edit

- `ours-direct`: Table1 fair (`edit_latent_source_similarity_rerank`)
- `ours-assisted`: Table1 attack (`edit_latent_table_success_rerank`)

### De novo

- `ours-one-shot`: direct-smiles v2 SFT / group RL
- `ours-assisted`: dualmode + `latent_property_rerank`

## 当前最值得优先回答的问题

1. OOD 最优主结果到底是 `group RL`、`conservative decoding`，还是两者叠加？
2. Table1 里真正来自生成模型本体的增益有多少，selection gain 有多少？
3. 我们能不能在外部 multi-property IND/OOD benchmark 上站住脚？
