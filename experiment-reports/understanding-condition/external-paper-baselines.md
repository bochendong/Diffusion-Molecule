# External Paper Baselines

| 字段 | 值 |
| --- | --- |
| **状态** | initial baseline extraction |
| **最后更新** | 2026-06-29 |
| **用途** | 把我们的 Table1 / MuMO / C-MuMO 结果放到外部论文口径下，而不是只和自己历史版本比较 |

## 为什么要单独建这张表

当前 MuMO assisted edit 的 `Sim >= 0.4` 只是内部 source-conditioned edit 诊断，不能当作外部 SOTA 对比。论文主表必须按对方论文的 official metrics 对齐：

1. MolEditRL / MolEdit-Instruct：`Validity`、`Acc_all` / `Acc_valid`、`FCD`。
2. GeLLM3O / MuMOInstruct：`Success Rate (SR)`、`Similarity`、`Relative Improvement (RI)`，每个 input 生成 20 个候选。
3. GeLLMO-C / C-MuMOInstruct：`SR`、`Similarity`、`RI`，另有 stricter success criteria。
4. C-MORAL：`SOR`、`SSOR`、`Sim`、`RI`，是 RL post-training 方向的重要新对照。

## Confirmed Baseline Targets

### MolEditRL / MolEdit-Instruct Table1

来源：MolEditRL Table1 screenshot，已整理在 [MolEditRL Reference Comparison](../moleditrl-baseline-comparison.md)。

| 对比项 | MolEditRL | 我们当前 assisted best | 备注 |
| --- | ---: | ---: | --- |
| overlap-3 mean `Acc_all(0.65)` | **0.555** | **0.860** | RB / MW / SA 三个重合单任务 |
| overlap-3 mean `Acc_all(0.15)` | **0.838** | **0.860** | 同上 |
| 10-task mean `Acc_all(0.65)` | **0.450** | **0.794** | 我们为 table-success rerank assisted result；旧报告 0.894 为算术错误 |
| 10-task mean `Acc_all(0.15)` | **0.727** | **0.854** | GSK3B 仍为 0 |

结论：Table1 是目前最容易写成“超过已有方法”的主表，但必须把 base generator、fair result、assisted/selection result 分开。

### GeLLM3O / MuMOInstruct

来源：GeLLM3O 论文与官方 repo。论文定义 `SR` 为每个 test input 的 20 个生成候选里至少一个分子能同时改善所有目标性质的比例；`Similarity` 是 optimized molecule 与 input molecule 的 Tanimoto；`RI` 是相对改善。

| 设置 | 外部 best / target | 我们当前状态 |
| --- | ---: | --- |
| IND average `SR` | **76.8%** / **76.1%**（best generalist GeLLM3O variants） | evaluator 已支持 candidate-level `SR`；待重跑 + oracle CSV |
| OOD average `SR` | **88.7%** / **90.8%**（best generalist GeLLM3O variants） | evaluator 已支持 candidate-level `SR`；待重跑 + oracle CSV |
| Similarity band | 约 **0.5-0.6** | 新 evaluator 输出 `Similarity(success)`；`Sim >= 0.4` 仅保留为内部诊断 |
| candidate budget | **20** per input | direct/group-RL 脚本已默认保存并评估 20-candidate CSV；assisted/rich/GraphEditDSL 需单独标 assisted |

目标：MuMO 主文表不能报 proxy/Sim 代替 SR。下一步必须补 ADMET/TDC generated-property oracle CSV，并重跑 direct/group-RL，得到 official-style SR / Sim / RI。

### GeLLMO-C / C-MuMOInstruct

来源：GeLLMO-C 论文与官方 repo。C-MuMO 是 property-specific objective benchmark，要求改善 sub-optimal properties，同时保持 near-optimal properties。

| 设置 | 外部 best / target | 我们当前状态 |
| --- | ---: | --- |
| IND average `SR` | 约 **74.8%**（best generalist row from main table） | evaluator 已支持；未跑 C-MuMO official oracle |
| OOD average `SR` | 约 **63.0%**（best generalist row from main table） | evaluator 已支持；未跑 C-MuMO official oracle |
| Strict setting | paper appendix reports stricter success criteria | 未实现同口径 |
| candidate budget | **20** per input | 我们需要用 n=20 direct + assisted 分表 |

目标：C-MuMO 是比 MuMO 更贴近“可控多目标”的外部表。我们需要先跑 official `NingLab/C-MuMOInstruct` test，并补 property oracle。

### C-MORAL / C-MuMO RL Post-training

来源：C-MORAL 论文。它不是和 GeLLMO-C 完全同一张 SR 表，而是 RL post-training 表，指标为 `SOR` / `SSOR` / `Sim` / `RI`。

| 方法 | IND SOR | IND SSOR | IND Sim | OOD SOR | OOD SSOR | OOD Sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GeLLM4O-C Mistral baseline | 33.7 | 14.4 | 0.58 | 24.4 | 9.8 | 0.60 |
| GeLLM4O-C Llama baseline | 30.2 | 13.7 | 0.56 | 25.2 | 10.2 | 0.57 |
| **C-MORAL GRPO Mistral** | **48.9** | **25.1** | 0.59 | **39.5** | **20.8** | 0.60 |
| C-MORAL GDPO Mistral | 47.0 | 25.0 | 0.59 | 38.3 | 19.8 | 0.59 |
| C-MORAL GRPO Llama | 40.3 | 20.1 | 0.58 | 37.2 | 19.6 | 0.60 |

目标：如果我们要讲 RL + agentic，C-MORAL 是必须引用的新线。短期先不用硬打 C-MORAL 的 SOR 表，但论文 related benchmark 里必须放它，后续若能接上 C-MuMO + RL reward，可作为强对照。

## Immediate Work Items

1. 从 GeLLM3O / GeLLMO-C paper HTML/PDF 里抽 task-level SR / Sim / RI，做 machine-readable CSV。
2. 给 MuMO / C-MuMO evaluator 补 official generated-property oracle CSV：BBBP、DRD2、HIA、Mutagenicity、hERG、Liver/DILI、PAMPA/AMPA、pLogP、QED 等。
3. 重新评估我们的 direct / agentic rich / GraphEditDSL 输出，输出 `SR`、`Sim(success)`、`RI(success)`，而不是只报 proxy 和 `Sim >= 0.4`。
4. 主文表按 line 分开：`ours-direct n=20`、`ours-assisted rich/GraphEditDSL`、`ours-RL/agentic`，避免把 assisted result 和 one-shot baselines 混表。
5. 继续保留内部 `Sim >= 0.4`，但只作为 source preservation diagnostic。

## External Sources

- MolEditRL: https://arxiv.org/abs/2505.20131
- GeLLM3O / MuMOInstruct: https://aclanthology.org/2025.acl-long.1225/
- GeLLMO official repo: https://github.com/ninglab/GeLLMO
- GeLLMO-C / C-MuMOInstruct: https://aclanthology.org/2025.findings-emnlp.1145/
- GeLLMO-C official repo: https://github.com/ninglab/GeLLMO-C
- C-MORAL: https://arxiv.org/html/2604.23061
