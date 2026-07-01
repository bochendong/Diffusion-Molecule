# Paper Benchmark Plan

| 字段 | 值 |
| --- | --- |
| **状态** | active planning |
| **最后更新** | 2026-06-29 |
| **项目** | `SketchMol-Understanding-Condition` |
| **目标** | 把当前结果整理成可投顶会的 benchmark 结构，并提供可直接提交到服务器的命令 |

外部论文 baseline 数值目标已单独整理到 [external-paper-baselines.md](external-paper-baselines.md)。MuMO / C-MuMO evaluator 已修复为 candidate-level official-style `SR` / `Similarity` / `RI` 聚合；当前 `Sim >= 0.4` 只作为内部 source-preservation diagnostic。下一步需要重跑 direct/group-RL，并补 ADMET/generated-property oracle CSV。

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

### 5. MuMO / C-MuMO benchmark port ✅ ADMET oracle SR 可报（MuMO）

目的：把我们的方法放到外部论文已经使用的 IND/OOD multi-property benchmark 上。

**Proxy-only diagnostic（缺 ADMET 前）**：

| Job | 变体 | top-20 SR | Sim≥0.4 | DPQ SR |
| --- | --- | ---: | ---: | ---: |
| `16997792` | MuMO GraphEdit 2-step + proposal | 1.35% | 46% | 13.5% |
| `17010445` | MuMO GraphEdit 1-step | 0.95% | 99.9%† | 9.5% |
| `17010444` | MuMO GraphEdit source-only | 0% | 99.4% | 0% |

† 1-step 候选池 Sim 虚高；selected-1 Sim **24.8%**。

**ADMET oracle-backed（`17047446` + re-eval）**：

| Job | 变体 | SR (all) | IND avg | OOD avg | Sim(success) |
| --- | --- | ---: | ---: | ---: | ---: |
| **`16997792`** | **MuMO GraphEdit 2-step** | **48.3%** | **28.1%** | **69.0%** | **0.19** |
| `17010444` | source-only | 8.1% | — | — | 0.51 |
| `17010445` | 1-step | 3.0% | — | — | 0.49 |
| `17010957` | C-MuMO 2-step | 0% | — | — | — |

Oracle CSV：`outputs/external_oracle_build_v1/generated_properties.csv`（39,522 SMILES，ADMET 97.1%）。GeLLMO 参考：IND **~77%**，OOD **~89%**。C-MuMO 待 source-side oracle（maintain/improve）。

**MuMO 对照（direct vs assisted 分开报）**：

| 变体 | 线 | Valid | Proxy | Sim ≥0.4 | SR |
| --- | --- | ---: | ---: | ---: | ---: |
| source-copy sanity | oracle | 100% | 46.7% | **100%** | — |
| one-shot direct（official `16894722`） | direct | 90.7% | 16.4%† | 0.05% | **0** |
| de novo group-RL direct | direct | 85.1% | 40.7%‡ | 0 | 0 |
| source-edit SFT direct | direct | 97.3% | 45.4%‡ | 0 | 0 |
| agentic rich v2 | assisted | 100% | 46.7% | 32.3% | 0 |
| GraphEditDSL heuristic 1-step | assisted | 100% | 46.7% | 26.2% | 0 |
| policy GraphEditDSL v2 | assisted | 100% | 46.7% | 19.7% | 0 |
| **GraphEditDSL heuristic 2-step** | assisted | **100%** | **46.7%** | **45.6%** | **0** |
| agentic rich x2 | assisted | 100% | 46.7% | 42.5% | 0 |

† candidate-level proxy；‡ 旧 selected-prediction 口径。上表 SR=0 列为 proxy-only 时代；**oracle-backed MuMO GraphEdit 2-step SR=48.3%** 见上表。

**MuMO 主结果（oracle-backed，`16997792`）**：GraphEdit 2-step top-20 SR **48.3%**（IND **28.1%** / OOD **69.0%**）；Sim(success) **0.19**。下一项：C-MuMO source oracle + RI 修复。

数据：`/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/{train,test}.json`（HuggingFace 官方）。

当前状态：

1. `adapter v1` 已实现，记录见 [external-multiproperty-benchmark.md](external-multiproperty-benchmark.md)。
2. 已覆盖 GeLLMO/MuMOInstruct 和 GeLLMO-C/C-MuMOInstruct 的 5 IND + 5 OOD task specs。
3. 已新增 source-conditioned CSV exporter、external generated-property evaluator、one-shot server submit 入口。
4. 默认 pilot 关闭 output-side property rerank，避免把外部 benchmark 第一版做成 assisted result。
5. 已新增 external source-conditioned group-RL 入口；reward 加 source Tanimoto 项，默认 `DISABLE_PROPERTY_RERANK=1`，用于主文 `ours-group-rl` 候选。
6. 已新增 `append_source_property_program` condition mode 和 MuMO source-edit SFT / SFT+group-RL 入口；生成主干显式看到 source SMILES，不依赖 output-side rerank。
7. 已新增 MuMO `agentic revise` 入口；它是 assisted/source-edit line，单独报，不与 one-shot direct generation 混表。
8. 已新增 MuMO `agentic revise rich` 入口；复用 v1 direct proposals，扩大 beam/candidate cap，并使用 similarity-first selection。
9. 已新增 MuMO `GraphEditDSL agent` 入口；planner 输出结构化 graph-edit actions，RDKit executor 在 source graph 上执行，verifier 按 local property success + source similarity 选候选。这是 source-conditioned edit 的主方向升级，不再只是 rerank/local-search 调参。
10. 已新增 MuMO `policy GraphEditDSL v2` 入口；2-step beam expansion + property-aware action scoring + richer DSL action set，用来判断 GraphEditDSL 是否能追平 rich v2。
11. 已新增 `submit_external_multiproperty_official_suite.sh`：一次提交 MuMO / C-MuMO source-copy、target-copy、GraphEditDSL top-20 candidate-level evaluation。
12. 已新增 `score_external_multiproperty_oracles.py`：合并外部 ADMET CSV，补本地/TDC 可算性质，并输出 official evaluator coverage report。

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

source-edit repair v1 提交命令（先 SFT，完成后再 RL）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft.sh

git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh
```

agentic revise v1 提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh
```

agentic revise rich v2 提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise_rich.sh
```

GraphEditDSL agent v1 提交命令（主方向升级，复用 v1 direct proposals）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_agent.sh
```

policy GraphEditDSL v2 提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_policy.sh
```

flight sweep 提交命令：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_flight_sweep.sh
```

MuMO / C-MuMO official suite 提交命令（推荐当前主线）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

如果已有外部 ADMET/generated-property CSV：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=/path/to/generated_properties.csv \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

注意：BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等性质需要 external generated-property CSV 才能公平评估；没有 oracle CSV 时只作为 coverage / plumbing pilot。

#### C-MuMO 数据下载（已完成 2026-07-01）

官方数据源：`NingLab/C-MuMOInstruct`（HuggingFace）。与 MuMO 不同，test split 按 task 拆成 10 个 `test_*_mixed.json.gz`；fetch 脚本会合并并规范化 `task` / `improved` / `stable` 字段。

```bash
bash SketchMol-Understanding-Condition/scripts/fetch_external_cmumo_dataset.sh
# 输出：/scratch/bdong/datasets/Diffusion-Molecule/external/cmumo/test.json
```

MuMO 如需重下：

```bash
bash SketchMol-Understanding-Condition/scripts/fetch_external_mumo_dataset.sh
```

C-MuMO official suite：

```bash
git pull --ff-only
SUCC_OFFICIAL_RUN_MUMO=0 \
SUCC_OFFICIAL_CMUMO_SOURCE_FILE=/scratch/bdong/datasets/Diffusion-Molecule/external/cmumo/test.json \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

#### ADMET generated-property CSV（GeLLMO 官方 predictor）

GeLLMO / GeLLMO-C 论文使用 **ADMET-AI** 作为 ADMET oracle（`process-output.ipynb` → `generate_props()` → `*-admet_props.csv`），再与本地/TDC 性质（QED、pLogP、SA、DRD2）合并（`evaluate.ipynb` → `compute_props()`）。

| GeLLMO ADMET-AI 列 | SUCC oracle 名 |
| --- | --- |
| `BBB_Martins` | `bbbp` |
| `HIA_Hou` | `hia` |
| `AMES` | `mutagenicity` |
| `hERG` | `erg` |
| `DILI` | `liver` |
| `PAMPA_NCATS` | `ampa` |
| `Carcinogens_Lagunin` | `carc` |

DRD2 仍走 TDC Oracle（与 GeLLMO `data/props/` 一致）；QED / pLogP / SA 由 `score_external_multiproperty_oracles.py` 本地补全。

安装 ADMET-AI（建议单独 venv；overlay 若无 `admet-ai` 则 `pip install admet-ai`）：

```bash
pip install admet-ai
```

从 prediction CSV 构建 generated-property oracle：

```bash
SUCC_ORACLE_INPUT_CSV=SketchMol-Understanding-Condition/outputs/external_mumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv \
SUCC_ORACLE_OUTPUT_CSV=SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv \
bash SketchMol-Understanding-Condition/scripts/run_external_multiproperty_generated_oracle_pipeline.sh
```

Slurm 提交：

```bash
SUCC_ORACLE_INPUT_CSV=.../graph_edit_agent_candidate_predictions.csv \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_generated_oracle_pipeline.sh
```

生成后重跑 official suite：

```bash
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

venv 安装（ComputeCanada，需先 `module load rdkit`）：

```bash
bash SketchMol-Understanding-Condition/scripts/setup_admet_ai_venv.sh
export SUCC_ADMET_PYTHON_BIN=$HOME/.venvs/admet_ai/bin/python
```

参考： [GeLLMO repo](https://github.com/ninglab/GeLLMO) · [GeLLMO-C repo](https://github.com/ninglab/GeLLMO-C) · [ADMET-AI](https://github.com/swansonk14/admet_ai)

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
3. 我们能不能在外部 multi-property IND/OOD benchmark 上站住脚？→ **heuristic GraphEditDSL 2-step Sim 45.6%**（assisted best）；rich x2 **42.5%**；direct 训练线仍 Sim≈0；official SR 待 oracle CSV。
