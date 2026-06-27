# Paper Benchmark Plan

| 字段 | 值 |
| --- | --- |
| **状态** | active planning |
| **最后更新** | 2026-06-27 |
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

## P0: 立刻要跑的实验（2026-06-27 已完成）

### 1. Table1 公平版重跑 ✅（jobs `16768165`–`16768167`）

目的：给当前最强的 v3 MolEdit 线补一条**不依赖 table-success rerank**的公平版结果。

结果位置：

- pack: `.../dataset/table1_benchmark_fair/`
- eval latent: `.../univideo_molecule/table1_eval_latent_fair/`
- benchmark: `.../univideo_molecule/benchmark_materialized_table1_fair/`
- metrics: `.../univideo_molecule/moledit_table_metrics_table1_fair/`

**结果**（`edit_latent_source_similarity_rerank`，10 tasks × 100，primary_fast）：

| 指标 | 值 |
| --- | ---: |
| mean `Acc_all(0.65)` | **0.286** |
| mean `Acc_all(0.15)` | **0.569** |
| Validity | 1.000（全 task） |
| 最差 task | GSK3B↑ `Acc_all(0.65)=0.000` |
| 最好 task | DRD2↓ MW↓ SA↓ `Acc_all(0.65)=0.490` |

与 `table_success_rerank` attack 线并排可量化 **selection gain**；公平版本体增益偏低，GSK3B↑ 仍为零。

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_v3_fair_table1_extension.sh
```

成功标准：

- 10-task coverage 保持
- 给出 `Acc_all(0.65)` / `Acc_all(0.15)` 的公平版
- 能和 `table_success_rerank` 结果并排，量化 selection gain

### 2. OOD group RL + conservative decoding（n=128）✅（job `16768168`）

目的：检验 group RL checkpoint 在 conservative decoding 下，是否能同时保持 OOD overall 提升并修复 7p bucket。

**结果**：overall strict **80.2%**（+4.6pp vs group RL default n=128）；7p **78.0%**（+19.0pp）；validity **97.0%**。**同时超过 conservative SFT**（74.1% / 72.0%）。✅ 全部成功标准达成。

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_benchmark.sh
```

成功标准：

- overall strict > `0.756`
- 7p bucket > `0.590`
- 最理想是同时超过 conservative SFT 的 `0.741 / 0.720`

### 3. OOD group RL n=256（default decoding）✅（job `16768169`）

目的：测试 OOD 线的 sample-budget ceiling。

**结果**：overall strict **89.0%**（+13.4pp vs n=128 group RL）；7p **90.0%**；validity **99.0%**。sample budget 是 OOD 主瓶颈之一。✅

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_n256_benchmark.sh
```

成功标准：

- overall strict > `0.756`
- validity 不显著掉
- 判断 OOD 的瓶颈更像 training 还是 decoding/sample budget

### 4. OOD group RL n=256 + conservative decoding ✅（job `16768170`）

目的：测试当前最有机会成为 OOD 主结果的组合。

**结果**：overall strict **89.4%**（**OOD overall best**）；7p **87.0%**；validity **98.9%**；2p **82.7%**。✅ 同时刷新 overall 与 7p（相对 n=128 baseline）。

提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_n256_benchmark.sh
```

成功标准：

- 同时刷新 OOD overall strict 和 7p bucket
- 如果这条线最好，则作为主文 OOD 主结果

## P1: 下一批要接的外部 benchmark

### 5. MuMO / C-MuMO benchmark port ✅ diagnostics 完成（`16774795` / `16779361` / `16779362`）

目的：把我们的方法放到外部论文已经使用的 IND/OOD multi-property benchmark 上。

**MuMO 三方对照**（见 [external-multiproperty-benchmark.md](external-multiproperty-benchmark.md)）：

| 变体 | Job | Proxy eval prop frac | Sim ≥0.4 | Valid |
| --- | --- | ---: | ---: | ---: |
| source-copy sanity | `16779362` | 46.7% | **100%** | 100% |
| one-shot baseline | `16779361` | **42.5%** | 0 | 90.7% |
| group-RL | `16774795` | 40.7% | 0 | 85.1% |

**结论**：source-copy Sim=100% 排除 eval 链路 bug；one-shot ≈ group-RL，RL 无增益；瓶颈是 **source-edit 能力**（de novo ckpt），非数据/评估问题。

数据：`/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/{train,test}.json`（HuggingFace 官方）。

当前状态：

1. `adapter v1` 已实现，记录见 [external-multiproperty-benchmark.md](external-multiproperty-benchmark.md)。
2. 已覆盖 GeLLMO/MuMOInstruct 和 GeLLMO-C/C-MuMOInstruct 的 5 IND + 5 OOD task specs。
3. 已新增 source-conditioned CSV exporter、external generated-property evaluator、one-shot server submit 入口。
4. 默认 pilot 关闭 output-side property rerank，避免把外部 benchmark 第一版做成 assisted result。
5. 已新增 external source-conditioned group-RL 入口；reward 加 source Tanimoto 项，默认 `DISABLE_PROPERTY_RERANK=1`，用于主文 `ours-group-rl` 候选。

预期产出：

- `outputs/mumo_port_v1/`
- `outputs/cmumo_port_v1/`
- `outputs/direct_smiles_external_multiproperty_group_rl_v1/`

one-shot 提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_one_shot_baseline.sh
```

source-copy sanity 提交命令（不作为方法结果，只验证 source similarity / evaluator）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_source_copy_sanity.sh
```

group-RL 提交命令（官方文件里含 train/test split）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SOURCE_FILE=/path/to/mumo_or_cmumo_all_splits.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
```

group-RL 提交命令（train/test 独立文件）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_SOURCE_FILE=/path/to/mumo_train.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_SOURCE_FILE=/path/to/mumo_test.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
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

1. ~~OOD 最优主结果到底是 `group RL`、`conservative decoding`，还是两者叠加？~~ → **conservative n=256 overall 89.4%**；7p peak 在 default n=256 **90.0%**。
2. Table1 里真正来自生成模型本体的增益有多少，selection gain 有多少？→ 公平版 mean Acc@0.65 **28.6%**；需与 attack 线并排。
3. 我们能不能在外部 multi-property IND/OOD benchmark 上站住脚？→ **MuMO pipeline 跑通**；source-copy 排除 eval bug；**source-edit 能力是主瓶颈**（需 SFT/agentic，非 RL 调参）。
