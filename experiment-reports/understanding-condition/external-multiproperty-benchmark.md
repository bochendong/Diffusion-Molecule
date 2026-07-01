# External Multi-property Benchmark Port

| 字段 | 值 |
| --- | --- |
| **状态** | **official suite 6/6 完成**；MuMO GraphEdit 2-step top-20 Sim **46%** / DPQ SR **13.5%**（proxy-only lower-bound）；**待 ADMET CSV** 才能报真实 SR |
| **最后更新** | 2026-07-01（official suite `16997790`–`17010957`） |
| **代码范围** | `SketchMol-Understanding-Condition` |
| **目标** | 把 SUCC direct-SMILES/LLM-conditioned 线接到 GeLLMO/MuMOInstruct 和 GeLLMO-C/C-MuMOInstruct 风格的 source-conditioned multi-property IND/OOD benchmark |

> 2026-07-01 infrastructure update：新增 official-style suite。GraphEditDSL 现在默认输出 **selected-1** 和 **top-20 candidate CSV**；evaluator 支持 C-MuMO `improve` / `maintain` objective；新增 oracle coverage builder，可合并外部 ADMET CSV 并补本地/TDC 可算性质。主报告应看 candidate-level `SR / Sim(success) / RI(success)`，`Sim>=0.4` 继续只作 source-preservation diagnostic。

## 关键判断

MuMO / C-MuMO 不是 zero-source absolute target design。它们的任务形式是：

1. 输入 source molecule。
2. 按任务要求 increase / decrease 多个性质。
3. 保持结构变化尽量小。
4. 对 5 IND + 5 OOD task 分别统计 multi-property success。

所以这一版不应该复用 2p/7p 的 zero-source CSV，而应该走 source-conditioned edit CSV。

## 已实现

新增 exporter：

```bash
SketchMol-Understanding-Condition/scripts/export_external_multiproperty_benchmark_rows.py
```

覆盖 task specs：

| Suite | IND tasks | OOD tasks |
| --- | --- | --- |
| MuMO | BDP, BDQ, BPQ, DPQ, BDPQ | MPQ, BDMQ, BHMQ, BMPQ, HMPQ |
| C-MuMO | BPQ, ELQ, ACEP, BDPQ, DHMQ | CDE, ABMP, BCMQ, BDEQ, HLMPQ |

新增 evaluator：

```bash
SketchMol-Understanding-Condition/scripts/evaluate_external_multiproperty_predictions.py
```

它支持两层评估：

1. 本地可算的 proxy properties：QED、LogP、SA、pLogP proxy。
2. 外部 generated-property CSV：用于 BBBP、HIA、mutagenicity、hERG、DILI、PAMPA 等 ADMET/oracle 性质。
3. official-style 聚合：按 `condition_id` 把同一 input 的多个候选合并，`SR` 表示 **任一候选** 满足所有 task properties；`Similarity` / `RI` 在成功 input 的最佳成功候选上统计。

注意：`Internal strict` 仍额外要求 `Sim >= 0.4`，只作为 source-preservation diagnostic；不能替代 GeLLM3O / GeLLMO-C 的 `SR`。

新增 server 入口：

```bash
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_benchmark.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_group_rl.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_one_shot_baseline.sh
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_multiproperty_source_edit_sft.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh
SketchMol-Understanding-Condition/scripts/build_external_agentic_revise_predictions.py
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_mumo_agentic_revise.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise_rich.sh
SketchMol-Understanding-Condition/scripts/build_external_graph_edit_agent_predictions.py
SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_mumo_graph_edit_agent.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_agent.sh
SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_policy.sh
SketchMol-Understanding-Condition/scripts/run_external_multiproperty_source_copy_sanity.sh
SketchMol-Understanding-Condition/scripts/submit_external_mumo_source_copy_sanity.sh
```

one-shot 默认是 pilot/diagnostic：

- `RUN_TRAIN=0`
- 使用现有 direct-SMILES checkpoint。
- `NUM_SAMPLES=20`，对齐 GeLLMO 论文推理设置。
- 默认额外输出 `direct_smiles_candidate_predictions.csv`，并用它评估 official-style `SR`。
- `DISABLE_PROPERTY_RERANK=1`，避免第一版变成 output-side rerank。
- 如果没有 generated-property CSV，报告会把 missing oracle properties 明确列出来。

group-RL 入口默认：

- 从当前最强的 direct-SMILES group-RL checkpoint warm start。
- 使用 source-conditioned external rows 训练；`SFT_WEIGHT=0.15`，避免 target placeholder 过度鼓励复制 source。
- reward 仍使用本地可算 proxy strict/distance，同时新增可关闭的 source Tanimoto reward（默认权重 `0.5`，阈值 `0.4`）。
- benchmark 阶段默认 `NUM_SAMPLES=20`、`DISABLE_PROPERTY_RERANK=1`。
- benchmark 阶段默认输出候选级 CSV，并以 candidate-level aggregation 评估 `SR / Similarity / RI`。
- ADMET/oracle 性质仍需要 generated-property CSV 才能公平报告；没有 oracle CSV 时，group-RL 只优化 proxy subset + source conservation。

## Slurm 运行记录

| Job | 名称 | 数据 | 输出 |
| --- | --- | --- | --- |
| `16774795` | `succ-external-mumo-group-rl` | MuMO train/test（HuggingFace 官方 JSON） | `direct_smiles_external_mumo_group_rl_v1/` |
| `16779361` | `succ-external-mumo-one-shot` | MuMO test（one-shot baseline） | `direct_smiles_external_mumo_one_shot_v1/` |
| `16779362` | `succ-external-mumo-source-copy` | MuMO test（source-copy sanity） | `external_mumo_source_copy_sanity/` |
| `16800837` | `succ-external-mumo-source-edit-sft` | MuMO train/test（source-edit SFT warm-start） | `direct_smiles_external_mumo_source_edit_sft_v1/` |
| `16806903` | `succ-external-mumo-source-edit-rl` | MuMO train/test（source-edit SFT → group-RL） | `direct_smiles_external_mumo_source_edit_sft_group_rl_v1/` |
| `16813212` | `succ-external-mumo-agentic-revise` | MuMO test（source-edit SFT proposal + 2-step local edit） | `direct_smiles_external_mumo_agentic_revise_v1/` |
| `16819986` | `succ-external-mumo-agentic-revise-4step` | MuMO test（复用 v1 proposals + 4-step local edit） | `direct_smiles_external_mumo_agentic_revise_4step_v1/` |
| `16824486` | `succ-external-mumo-agentic-rich` | MuMO test（rich actions + 2048 candidates/row） | `direct_smiles_external_mumo_agentic_revise_rich_v1/` |
| `16825306` | `succ-external-mumo-graph-edit` | MuMO test（GraphEditDSL planner + RDKit executor） | `direct_smiles_external_mumo_graph_edit_agent_v1/` |
| `16892025` | `succ-external-mumo-graph-policy` | MuMO test（policy GraphEditDSL 2-step planner） | `direct_smiles_external_mumo_graph_edit_policy_v2/` |
| `16894722` | `succ-external-mumo-one-shot` | MuMO test（official SR evaluator 重跑） | `direct_smiles_external_mumo_one_shot_v1/` |

集群数据路径（已下载）：

- train：`/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json`（828k rows）
- test：`/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json`（44k rows）

## MuMO group-RL pilot（job `16774795`）

入口：`submit_direct_smiles_external_multiproperty_group_rl.sh`（`16742f8`）。MuMO official train/test；10 tasks × 200 train rows（2000 total）；eval 1992 rows（HMPQ 192）；warm-start 2p7p group-RL ckpt；1 epoch group-relative RL（`SFT_WEIGHT=0.15`，`source_similarity_weight=0.5`）；benchmark n=20，`DISABLE_PROPERTY_RERANK=1`；**无 generated-property oracle CSV**。

训练：`eval_mean_reward` **-0.337**；`mean_reward` **-0.660**；`sft_loss` **1.964**。

评测（proxy-only；strict 需 oracle CSV 才能算）：

| Split | Valid | Proxy eval prop frac | Strict | Sim ≥0.4 |
| --- | ---: | ---: | ---: | ---: |
| IND（5 tasks） | **83.6%** | **42.2%** | 0 | 0 |
| OOD（5 tasks） | **86.8%** | **39.3%** | 0 | 0 |
| **all** | **85.1%** | **40.7%** | **0** | **0** |

分 task proxy eval prop frac（最高 / 最低）：

| Task | Split | Valid | Proxy eval prop frac |
| --- | --- | ---: | ---: |
| BPQ | ind | 93.5% | **62.3%** |
| MPQ | ood | 95.5% | **63.7%** |
| BDQ | ind | 69.5% | **23.2%** |
| BHMQ | ood | 71.0% | **17.8%** |

**结论**（初版；见下方 diagnostics 更新）：
1. **端到端 pipeline 跑通**：export → feature → group-RL → benchmark；10-task coverage OK。
2. **无 oracle CSV 时 strict=0 是预期行为**（BBBP/DRD2/HIA/mutagenicity 缺 external scorer）；proxy subset `eval prop frac` **~41%** 仅作 plumbing 诊断。
3. **source similarity 全为 0**（Sim ≥0.4）：初判为 source-edit 能力问题；已由 source-copy sanity 排除 eval 链路 bug。

## MuMO diagnostics（jobs `16779361` / `16779362`）

同口径 eval：MuMO test，10 tasks × 200，n=20，无 rerank，无 oracle CSV。

| Job | 变体 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | --- | ---: | ---: | ---: | ---: |
| `16779362` | **source-copy sanity** | **100%** | 46.7% | **100%** | 0 |
| `16779361` | **one-shot baseline** | **90.7%** | **42.5%** | **0** | 0 |
| `16774795` | group-RL | 85.1% | 40.7% | 0 | 0 |

**诊断结论**：
1. **source-copy Sim@0.4 = 100%** → `source_smiles` 字段与 Tanimoto evaluator **正常**；group-RL / one-shot 的 Sim=0 **不是数据或评估链路 bug**。
2. **one-shot proxy 42.5% ≈ group-RL 40.7%** → 短 RL 未带来增益，validity 还略降（90.7% vs 85.1%）；瓶颈在 **de novo ckpt 不具备 source-conditioned edit 能力**，而非 RL 超参。
3. **strict 全为 0**（三方一致）→ 仍需 ADMET/TDC generated-property CSV 才能报 official strict；source-copy strict=0 也符合预期（复制 source 不改善性质）。
4. **下一步**：先跑 source-edit SFT warm-start，再用该 checkpoint 接 source-edit group-RL；ADMET oracle 回填；C-MuMO 复跑。

> **口径更新（`13f3bec`）**：下方 `16779361` 等旧 job 使用 **selected-prediction** 聚合；`16894722` 起 direct 线改用 **candidate-level official SR** 聚合（n=20/input）。旧 proxy **42.5%** 与新版 candidate-level proxy **16.4%** 不可直接横比。

## MuMO one-shot official SR 重跑（job `16894722`）

入口：`submit_direct_smiles_external_mumo_one_shot_baseline.sh`（`13f3bec` 后重跑）。MuMO test 1992 inputs × n=20 candidates = **39840** candidate rows；de novo ckpt；无 rerank，无 oracle CSV。耗时 **~31m**。输出 `direct_smiles_external_mumo_one_shot_v1/benchmark_external_multiproperty/direct_smiles_candidate_predictions.csv`。

评测（**official candidate-level 口径**）：

| 指标 | 值 | 说明 |
| --- | ---: | --- |
| Valid（candidate-level） | **90.7%** | 与旧 selected `16779361` 一致 |
| Proxy eval prop frac | **16.4%** | candidate-level 均值；≠ 旧 selected **42.5%** |
| **SR** | **0** | 无 oracle CSV（`lower_bound_missing_oracle`） |
| Sim≥0.4（input diagnostic） | **0.05%** | 与旧 selected Sim=0 一致 |
| Internal strict | **0** | source-preservation diagnostic |

分 split Sim≥0.4：IND **0%**；OOD **0%**（BPQ 有 0.5% candidate-level hit）。

**结论**：新 evaluator 确认 direct one-shot **不具备 source-edit 能力**（Sim≈0）；SR 仍待 oracle CSV。旧 `16779361` 的 proxy/Sim 仅作 plumbing 参考，正式对外报数用本 job。

## Source-edit repair v1

当前 direct-SMILES checkpoint 是 zero-source de novo 训练出来的；MuMO/C-MuMO 则要求输入 source molecule 后做小改动。因此这版先修生成主干，而不是继续调 rerank：

1. 新增 `append_source_property_program` condition mode：在原 MLLM feature / fallback numeric feature 后，显式追加 source-SMILES token features，再追加 property program tokens。模型结构不变，旧 checkpoint 可 warm start。
2. 新增 source-edit SFT warm-start：默认从最强 2p7p group-RL checkpoint 起步，使用 MuMO train/test rows，关闭 property rerank，先把 Sim≥0.4 拉回来。
3. 新增 SFT checkpoint 上的 source-edit group-RL：使用同一 source-aware condition mode，加大 source-similarity reward，仍关闭 property rerank，目标是同时提升 source similarity 与 proxy property success。

这条线和 de novo 2p-7p/OOD 主线互不覆盖；Table1 edit 主链也不改。

## MuMO source-edit SFT（job `16800837`）

入口：`submit_direct_smiles_external_mumo_source_edit_sft.sh`（`e7b755e`）。从 2p7p group-RL ckpt warm-start；`append_source_property_program` + `reset_training_state=1`；MuMO train 2000 / eval 1992 rows；1 epoch SFT；benchmark n=20，无 rerank，无 oracle CSV。输出 `direct_smiles_external_mumo_source_edit_sft_v1/`。

训练：1 epoch，`eval_loss` **1.275**；`eval_perplexity` **3.58**；`reset_training_state` **true**。

评测（同 diagnostics 口径）：

| 变体 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | ---: | ---: | ---: | ---: |
| one-shot（`16779361`） | 90.7% | 42.5% | 0 | 0 |
| group-RL de novo（`16774795`） | 85.1% | 40.7% | 0 | 0 |
| **source-edit SFT（`16800837`）** | **97.3%** | **45.4%** | **0** | 0 |
| source-copy sanity（`16779362`） | 100% | 46.7% | **100%** | 0 |

**结论**：source-aware conditioning + 1 epoch MuMO SFT **提升了 validity（+6.6pp vs one-shot）和 proxy（+2.9pp）**，但 **Sim@0.4 仍为 0**——模型仍未学会保持 source scaffold。1 epoch / 2000 rows 可能不够；下一步接 **source-edit group-RL**（加大 `source_similarity_weight=2.0`）验证 RL 能否拉回 Sim。

## MuMO source-edit group-RL（job `16806903`）

入口：`submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh`。从 source-edit SFT ckpt（`16800837`）warm-start；`append_source_property_program`；MuMO train 2000 / eval 1992 rows；1 epoch group-relative RL（`SFT_WEIGHT=0.5`，`source_similarity_weight=2.0`，`source_similarity_threshold=0.4`）；benchmark n=20，无 rerank，无 oracle CSV。输出 `direct_smiles_external_mumo_source_edit_sft_group_rl_v1/`。

训练：`eval_mean_reward` **-0.791**；`mean_reward` **-0.874**；`sft_loss` **0.992**；`pg_loss` **-0.058**。

评测（同 diagnostics 口径）：

| 变体 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | ---: | ---: | ---: | ---: |
| one-shot（`16779361`） | 90.7% | 42.5% | 0 | 0 |
| group-RL de novo（`16774795`） | 85.1% | 40.7% | 0 | 0 |
| source-edit SFT（`16800837`） | 97.3% | 45.4% | 0 | 0 |
| **source-edit group-RL（`16806903`）** | **96.4%** | **45.0%** | **0** | **0** |
| source-copy sanity（`16779362`） | 100% | 46.7% | **100%** | 0 |

分 split：

| Split | Valid | Proxy eval prop frac | Sim ≥0.4 |
| --- | ---: | ---: | ---: |
| IND（5 tasks） | 95.1% | 41.3% | 0 |
| OOD（5 tasks） | 97.7% | 48.9% | 0 |

**结论**：在 SFT 基础上加大 source-similarity reward（权重 2.0）**未能拉回 Sim@0.4**；validity 略降（-0.9pp），proxy 基本持平（-0.4pp）。说明瓶颈不在 RL 超参，而在 **source-edit 监督/数据量不足**（1 epoch × 2000 rows）或 **生成范式**（direct SMILES 从零解码 vs 显式 edit/revise）。下一步优先：**加长 SFT epoch / 扩大 train rows**、ADMET oracle CSV 回填 strict、或接 **agentic revise loop**。

## MuMO agentic revise v1（job `16813212`）

入口：`submit_direct_smiles_external_mumo_agentic_revise.sh`（`0bc748d`）。MuMO test 1992 rows；warm-start source-edit SFT ckpt（`16800837`）；`append_source_property_program`；direct proposal n=20；**2-step** source-preserving local edit（beam=48，max 256 candidates/row）；无 property rerank，无 oracle CSV。耗时 **~1h48m**。输出 `direct_smiles_external_mumo_agentic_revise_v1/`。

Revise 统计：`mean_candidate_count` **256**；`mean_local_success_fraction` **96.5%**；`mean_source_tanimoto` **0.243**；`source_similarity_success_rate` **15.6%**。

评测（**assisted/source-edit 线**，不与 direct one-shot 混表）：

| 变体 | 线 | Valid | Proxy eval prop frac | Sim ≥0.4 | Strict |
| --- | --- | ---: | ---: | ---: | ---: |
| one-shot direct（`16779361`） | direct | 90.7% | 42.5% | 0 | 0 |
| source-edit SFT direct（`16800837`） | direct | 97.3% | 45.4% | 0 | 0 |
| source-edit group-RL direct（`16806903`） | direct | 96.4% | 45.0% | 0 | 0 |
| **agentic revise 2-step（`16813212`）** | **assisted** | **100%** | **46.7%** | **15.6%** | **0** |
| source-copy sanity（`16779362`） | oracle | 100% | 46.7% | **100%** | 0 |

分 split（agentic）：

| Split | Valid | Proxy eval prop frac | Sim ≥0.4 |
| --- | ---: | ---: | ---: |
| IND（5 tasks） | 100% | 50.0% | 15.8% |
| OOD（5 tasks） | 100% | 43.3% | 15.4% |

分 task Sim ≥0.4（agentic，最高 / 最低）：BDMQ **27.5%** / HMPQ **2.6%**。

**结论**：显式 local-edit revise **首次把 MuMO Sim@0.4 从 0 拉到 15.6%**，validity 达 **100%**，proxy 略升（+1.3pp vs SFT direct）。但距 source-copy **100%** 和 official strict 仍远；说明 **direct 解码不是 source-edit 的正确范式**，assisted revise 方向正确。

## MuMO agentic revise 4-step（job `16819986`）

入口：同 `submit_direct_smiles_external_mumo_agentic_revise.sh`，`SUCC_EXTERNAL_AGENTIC_STEPS=4`；**复用 2-step 的 direct proposals**（`RUN_DIRECT=0`）；其余超参与 2-step 相同（beam=48，max 256 candidates/row）。耗时 **~1h23m**。输出 `direct_smiles_external_mumo_agentic_revise_4step_v1/`。

| 变体 | Steps | Valid | Proxy | Sim ≥0.4 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| agentic revise 2-step（`16813212`） | 2 | 100% | 46.7% | **15.6%** | 0 |
| agentic revise 4-step（`16819986`） | 4 | 100% | 46.7% | **15.6%** | 0 |

Revise 统计与 2-step **完全一致**：`mean_candidate_count` **256**；`mean_source_tanimoto` **0.243**；`source_similarity_success_rate` **15.6%**；逐 task Sim 数值相同。

**结论**：在相同 beam / candidate cap（256/row）下，**4-step 无 Sim/proxy 增益**——瓶颈不在 edit depth，而在 **candidate pool 容量** 或 **local edit action 覆盖面**（terminal add/replace/remove 不足以逼近 source scaffold）。下一步优先：**扩大 beam / max_candidates**、 richer edit actions、ADMET oracle CSV 回填 strict。

## MuMO agentic revise rich v2（job `16824486`）

入口：`submit_direct_smiles_external_mumo_agentic_revise_rich.sh`（`1b67382`）。复用 v1 direct proposals；`edit_action_profile=rich`（fragment attach、bond-order edit、Br/S 等）；`beam=128`，`max_candidates_per_row=2048`；`selection_mode=similarity_first`。耗时 **~10h27m**。输出 `direct_smiles_external_mumo_agentic_revise_rich_v1/`。

Revise 统计：`mean_candidate_count` **2048**；`mean_local_success_fraction` **98.0%**；`mean_source_tanimoto` **0.347**；`source_similarity_success_rate` **32.3%**。

| 变体 | 线 | Valid | Proxy | Sim ≥0.4 | Strict |
| --- | --- | ---: | ---: | ---: | ---: |
| agentic 2-step（`16813212`） | assisted | 100% | 46.7% | 15.6% | 0 |
| agentic 4-step（`16819986`） | assisted | 100% | 46.7% | 15.6% | 0 |
| **agentic rich v2（`16824486`）** | **assisted** | **100%** | **46.7%** | **32.3%** | **0** |

分 split（rich v2）：IND Sim **34.9%**；OOD Sim **29.6%**。分 task Sim 最高 BDP **73.0%** / BDMQ **57.5%**；最低 HMPQ **5.7%**。

**结论**：rich actions + 2048 candidates/row + similarity-first **把 Sim@0.4 从 15.6% 拉到 32.3%（+16.7pp）**，proxy 不变；验证了 4-step 诊断——瓶颈在 **edit action 覆盖面与 candidate pool**，不在 step depth。

## MuMO GraphEditDSL agent v1（job `16825306`）

入口：`submit_direct_smiles_external_mumo_graph_edit_agent.sh`（`6d2b2b7`）。复用 v1 direct proposals；`planner_mode=heuristic_graph_dsl`；`site_limit=32`，`max_plans_per_property=160`，`max_candidates_per_row=4096`；RDKit executor + `similarity_first` verifier。耗时 **~22m**。输出 `direct_smiles_external_mumo_graph_edit_agent_v1/`（含 `graph_edit_plans.jsonl`）。

Agent 统计：`mean_plan_count` **441**；`mean_candidate_count` **89**；`mean_local_success_fraction` **91.1%**；`mean_source_tanimoto` **0.276**；`source_similarity_success_rate` **26.2%**。

| 变体 | 线 | Valid | Proxy | Sim ≥0.4 | Strict |
| --- | --- | ---: | ---: | ---: | ---: |
| agentic rich v2（`16824486`） | assisted | 100% | 46.7% | **32.3%** | 0 |
| **GraphEditDSL agent（`16825306`）** | **assisted** | **100%** | **46.7%** | **26.2%** | **0** |

分 split（GraphEditDSL）：IND Sim **27.0%**；OOD Sim **25.4%**。分 task Sim 最高 BDP **55.0%** / BDMQ **48.0%**；最低 HMPQ **2.6%**。

**结论**：显式 graph-edit action 空间 **有效**（26.2% > 15.6% baseline），但当前 **heuristic planner 仍低于 rich local revise（32.3%）**——candidate 利用率低（89 vs 2048/row）。GraphEditDSL 架构方向正确，下一步换 **LLM/policy planner**、扩大 executor candidate yield，而非继续加 local search step。

## MuMO policy GraphEditDSL v2（job `16892025`）

入口：`submit_direct_smiles_external_mumo_graph_edit_policy.sh`（`17079b0`）。复用 v1 direct proposals；`planner_mode=policy_graph_dsl`；2-step beam（beam=96）；richer DSL actions；`max_candidates_per_row=2048`；`similarity_first`。耗时 **~5h42m**。输出 `direct_smiles_external_mumo_graph_edit_policy_v2/`。

Agent 统计：`mean_candidate_count` **2046**（v1 heuristic **89**）；`mean_executed_plan_count` **4383**；`mean_plan_count` **15325**；`mean_source_tanimoto` **0.280**；`source_similarity_success_rate` **19.7%**。

| 变体 | 线 | Valid | Proxy | Sim ≥0.4 | SR |
| --- | --- | ---: | ---: | ---: | ---: |
| agentic rich v2（`16824486`） | assisted | 100% | 46.7% | **32.3%** | 0 |
| GraphEditDSL heuristic（`16825306`） | assisted | 100% | 46.7% | 26.2% | 0 |
| **policy GraphEditDSL v2（`16892025`）** | **assisted** | **100%** | **46.7%** | **19.7%** | **0** |

分 split（policy）：IND Sim **18.4%**；OOD Sim **20.9%**。分 task Sim 最高 BDMQ **53.0%** / BDP **20.5%**；最低 HMPQ **1.0%**。

**结论**：policy planner **把 candidate yield 从 89 拉到 2046/row（目标达成）**，但 Sim@0.4 **反而低于 heuristic（19.7% vs 26.2%）**，仍远低于 rich v2 **32.3%**——property-aware scoring 可能过度牺牲 source similarity。GraphEditDSL 主线仍成立，但下一步应调 **similarity/property 平衡** 或换 **LLM planner**，而非继续扩 beam。

## MuMO flight sweep 结果（`296be91`）

6/6 完成。SR 全线 **0**（无 oracle CSV）。Sim≥0.4 为 source-preservation diagnostic。

### Assisted / search 线

| Job | 变体 | Sim≥0.4 | Proxy | Valid | 耗时 | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `16824486` | rich v2（2048/row） | 32.3% | 46.7% | 100% | ~10h27m | — |
| **`16946788`** | **rich x2（4096/row）** | **42.5%** | 46.7% | 100% | ~1d5h | **+10.2pp vs rich v2** |
| **`16946787`** | **heuristic GraphEdit 2-step** | **45.6%** | 46.7% | 100% | ~12h04m | **assisted best**（+3.1pp vs rich x2） |

rich x2 统计：`mean_candidate_count` **4096**；`mean_source_tanimoto` **0.401**；IND Sim **44.9%**；OOD Sim **40.1%**；最高 task BDP **86%** / BDMQ **65.5%**。

**结论**：4096 candidates/row **有效**（32.3% → **42.5%**，+10.2pp），但 **仍略低于 GraphEditDSL 2-step（45.6%）**；且耗时 ~29h vs ~12h。candidate cap 不是唯一瓶颈——**graph-edit action 空间**仍略优。

### Direct 训练线（official candidate-level）

| Job | 变体 | Sim≥0.4 | Proxy | Valid | 耗时 | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `16800837` | SFT 1 epoch（旧口径） | 0 | 45.4%‡ | 97.3% | — | selected 口径 |
| `16946790` | **SFT long**（3 epoch × 400 rows/task） | **0.08%** | 25.7% | 100% | ~2h13m | 加长 SFT **未**拉回 Sim |
| `16946791` | **RL official 重跑**（n=64 cand） | **0.10%** | 15.8% | 99.6% | ~1h38m | candidate 口径仍 Sim≈0 |
| `16946792` | **high-sim RL**（sim_weight=8） | **0.05%** | 12.2% | 99.2% | ~3h50m | 高 sim reward **无效**，proxy 还降 |

**结论**：direct source-edit 训练（加长 SFT / 高 sim RL）**无法**恢复 source conservation；MuMO 主方向应押 **GraphEditDSL assisted** 而非继续训 direct decoder。

## Official MuMO / C-MuMO SR suite（`ec519ec`）

新增入口：

```bash
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

| Job | Suite | 线 | 输出 | 状态 |
| --- | --- | --- | --- | --- |
| `16997790` | MuMO | source/target-copy sanity | `external_mumo_official_copy_sanity_v1/` | **完成**（~1m） |
| `16997792` | MuMO | GraphEditDSL heuristic 2-step top-20 | `external_mumo_official_graph_edit_heuristic_2step_v1/` | **完成** |
| `17010444` | MuMO | GraphEditDSL **source-only** 2-step | `external_mumo_official_graph_edit_source_only_2step_v1/` | **完成** |
| `17010445` | MuMO | GraphEditDSL **1-step** top-20 | `external_mumo_official_graph_edit_1step_top20_v1/` | **完成** |
| `17010954` | C-MuMO | source/target-copy sanity | `external_cmumo_official_copy_sanity_v1/` | **完成** |
| `17010957` | C-MuMO | GraphEditDSL 2-step top-20 | `external_cmumo_official_graph_edit_heuristic_2step_v1/` | **完成** |

### Official suite 结果汇总（2026-07-01，**无 ADMET oracle CSV**）

**口径**：candidate-level top-20 `SR` / `Sim≥0.4`；`Status=official` 仅 **DPQ**（IND，drd2+plogp+qed 可本地/TDC 算）。其余 task 缺 BBBP/HIA/mutagenicity 等 → aggregate SR 是 **lower-bound**，不能和 GeLLMO 论文横比。`RI(success)` 在 proxy-only 成功样本上数值异常，暂不报。

#### MuMO GraphEdit ablation（top-20 candidate-level）

| Job | 变体 | SR (all) | Sim≥0.4 (all) | DPQ SR | DPQ Sim≥0.4 |
| --- | --- | ---: | ---: | ---: | ---: |
| `16997792` | 2-step + direct proposal | **1.35%** | **46.0%** | **13.5%** | 42.5% |
| `17010445` | 1-step top-20 | 0.95% | 99.9% | 9.5% | 99.5% |
| `17010444` | source-only 2-step | 0% | 99.4% | 0% | 97.5% |

Selected-1 diagnostic（与 flight sweep Sim 口径一致）：

| Job | SR (all) | Sim≥0.4 (all) | DPQ SR |
| --- | ---: | ---: | ---: |
| `16997792` | 0.60% | **45.6%** | 6.0% |
| `17010445` | 0.65% | 24.8% | 6.5% |
| `17010444` | 0% | 97.7% | 0% |

**Ablation 结论**：

1. **2-step beam expansion**：top-20 池在 DPQ 上 SR **13.5% vs 9.5%**（+4pp）；selected-1 Sim **45.6% vs 24.8%**。2-step 用更大搜索换更高 source-edit 质量 + 候选池 property 命中。
2. **Direct proposal 依赖**：source-only 2-step Sim≈99% 但 **DPQ SR=0**——GraphEditDSL 单独几乎只做 source 保持，不改善性质；需 direct/agentic proposal 作起点。
3. **1-step 候选池 Sim 虚高**：top-20 层 Sim≥0.4≈100%，但 selected-1 仅 24.8%——说明 1-step 生了很多近 source 候选，rerank/selected 路径与 top-20 oracle 口径不一致。

#### C-MuMO（`17010954` / `17010957`）

| Job | 线 | Sim≥0.4 (all) | SR (all) |
| --- | --- | ---: | ---: |
| `17010954` | copy sanity | 100% | 0 |
| `17010957` | GraphEdit 2-step top-20 | 99.95% | 0 |

C-MuMO plumbing OK（copy Sim=100%）；GraphEdit 候选池 Sim 极高但 **SR=0**——缺 ampa/bbbp/carc/erg/hia/liver/mutagenicity oracle（eval prop frac **42%**）。需 ADMET-AI CSV + 重跑。

**下一步（阻塞真实 SR）**：`17047446` ADMET-AI oracle build 运行中 → 完成后用 `generated_properties.csv` 重跑 official suite。

默认提交 4 条：

| Suite | 线 | 输出目录 | 说明 |
| --- | --- | --- | --- |
| MuMO | source-copy + target-copy sanity | `outputs/external_mumo_official_copy_sanity_v1/` | 校验 source similarity、target/oracle 上界 |
| MuMO | GraphEditDSL heuristic 2-step top-20 | `outputs/external_mumo_official_graph_edit_heuristic_2step_v1/` | 复用 MuMO direct proposal CSV，candidate-level official SR |
| C-MuMO | source-copy + target-copy sanity | `outputs/external_cmumo_official_copy_sanity_v1/` | 校验 C-MuMO objective / oracle 映射 |
| C-MuMO | GraphEditDSL heuristic 2-step top-20 | `outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/` | source-only GraphEditDSL 起步，不依赖 MuMO proposal CSV |

关键产物：

- `benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv`：每个 input 的 top-20 candidates。
- `benchmark_graph_edit_agent/external_multiproperty_report.md`：candidate-level official-style `SR / Sim(success) / RI(success)`。
- `benchmark_graph_edit_agent/selected_prediction_eval/external_multiproperty_report.md`：selected-1 diagnostic。
- `external_oracle_properties.report.md`：oracle property coverage；若 ADMET 缺失，不能 claim official SR。

如果已有外部 ADMET/generated-property CSV：

```bash
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=/path/to/generated_properties.csv \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

如果只想先跑 C-MuMO：

```bash
SUCC_OFFICIAL_RUN_MUMO=0 \
SUCC_OFFICIAL_CMUMO_SOURCE_FILE=/scratch/bdong/datasets/Diffusion-Molecule/external/cmumo/test.json \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

注意：suite 默认 `SUCC_OFFICIAL_BUILD_ORACLE_CSV=1`，会生成 coverage CSV/report；但 BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等仍需要外部 predictor CSV 才能达到完整 official coverage。

## MuMO flight sweep 提交记录

目的：用户长时间离线时一次性覆盖几条差异足够大的方向。

| Job | 脚本 | 方向 | 输出 |
| --- | --- | --- | --- |
| `16946786` | `graph_edit_similarity_anchor` | policy GraphEditDSL similarity-anchor | `graph_edit_policy_sim_anchor_v1/` |
| `16946787` | `graph_edit_heuristic_2step` | heuristic GraphEditDSL 2-step + beam 128 | `graph_edit_heuristic_2step_v1/` |
| `16946788` | `agentic_revise_rich_x2` | rich revise 4096 candidates/row | `agentic_revise_rich_x2_v1/` |
| `16946790` | `source_edit_sft_long` | 3 epoch SFT × 400 rows/task | `source_edit_sft_long_v1/` |
| `16946791` | `source_edit_rl_official_rerun` | source-edit RL official candidate-level 重跑 | `source_edit_rl_official_rerun_v1/` |
| `16946792` | `source_edit_rl_highsim` | high-sim group-RL（sim_weight=8, sft=1.0） | `source_edit_rl_highsim_v1/` |

一键提交：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_flight_sweep.sh
```

如果要轻量版，只跑 assisted/search 线：

```bash
git pull --ff-only
SUCC_FLIGHT_SOURCE_SFT_LONG=0 \
SUCC_FLIGHT_SOURCE_RL_OFFICIAL=0 \
SUCC_FLIGHT_SOURCE_RL_HIGHSIM=0 \
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_flight_sweep.sh
```

如果要只跑训练线：

```bash
git pull --ff-only
SUCC_FLIGHT_GRAPH_SIM=0 \
SUCC_FLIGHT_GRAPH_HEUR2=0 \
SUCC_FLIGHT_RICH_X2=0 \
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_flight_sweep.sh
```

## 服务器命令

MuMO one-shot baseline（同口径对照，推荐先跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_one_shot_baseline.sh
```

输出重点：

- selected prediction：`benchmark_external_multiproperty/direct_smiles_predictions.csv`
- candidate prediction：`benchmark_external_multiproperty/direct_smiles_candidate_predictions.csv`
- official-style report：`benchmark_external_multiproperty/external_multiproperty_report.md`
- summary CSV：`benchmark_external_multiproperty/external_multiproperty_summary.csv`

MuMO source-copy sanity（不算模型结果，只验证 source similarity / evaluator）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_source_copy_sanity.sh
```

最小 one-shot pilot（自定义数据路径）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/mumo_or_cmumo_test.json \
SUCC_EXTERNAL_MULTIPROP_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

如果已经有外部 ADMET/generated-property CSV，则加：

```bash
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=/path/to/generated_properties.csv \
SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV=/path/to/source_properties.csv \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

group-RL 公平版（同一个官方文件里含 train/test split）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SOURCE_FILE=/path/to/mumo_or_cmumo_all_splits.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK=200 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
```

group-RL 公平版（train/test 是两个独立文件）：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_SOURCE_FILE=/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_SOURCE_FILE=/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK=200 \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_group_rl_v1 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
```

group-RL 若要用外部 oracle CSV：

```bash
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_GENERATED_PROPERTIES_CSV=/path/to/generated_properties.csv \
SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SOURCE_PROPERTIES_CSV=/path/to/source_properties.csv \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_group_rl.sh
```

MuMO source-edit SFT warm-start（推荐下一条先跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft.sh
```

MuMO source-edit SFT + group-RL（等上一条完成后跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_source_edit_sft_group_rl.sh
```

MuMO agentic revise 2-step：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh
```

MuMO agentic revise 4-step（复用 2-step proposals，只跑 revise）：

```bash
git pull --ff-only
SUCC_EXTERNAL_AGENTIC_STEPS=4 \
SUCC_EXTERNAL_AGENTIC_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_4step_v1 \
SUCC_EXTERNAL_AGENTIC_SLURM_JOB_NAME=succ-external-mumo-agentic-revise-4step \
SUCC_EXTERNAL_AGENTIC_RUN_DIRECT=0 \
SUCC_EXTERNAL_AGENTIC_RUN_FEATURE_EXPORT=0 \
SUCC_EXTERNAL_AGENTIC_DIRECT_PREDICTION_CSV=SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1/direct_smiles_proposals.csv \
SUCC_EXTERNAL_AGENTIC_ROWS_CSV=SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1/external_multiproperty_rows.csv \
SUCC_EXTERNAL_AGENTIC_TASK_SPEC_JSON=SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_agentic_revise_v1/external_multiproperty_task_specs.json \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise.sh
```

MuMO agentic revise rich v2（推荐下一条跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_agentic_revise_rich.sh
```

MuMO GraphEditDSL agent v1（主方向升级，复用 v1 direct proposals）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_agent.sh
```

MuMO policy GraphEditDSL v2（推荐下一条跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_mumo_graph_edit_policy.sh
```

MuMO flight sweep（推荐长时间离线时跑）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_flight_sweep.sh
```

只跑 MuMO OOD：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/mumo_test.json \
SUCC_EXTERNAL_MULTIPROP_SUITE=mumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=ood \
SUCC_EXTERNAL_MULTIPROP_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/external_mumo_ood_pilot \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

如果已经有 ADMET/TDC generated-property CSV：

```bash
git pull --ff-only
SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE=/path/to/cmumo_test.json \
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=/path/to/generated_admet_props.csv \
SUCC_EXTERNAL_MULTIPROP_SUITE=cmumo \
SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT=all \
SUCC_EXTERNAL_MULTIPROP_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/external_cmumo_pilot \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_external_multiproperty_benchmark.sh
```

## 当前限制

这版是 benchmark port，不是最终 claim。

1. BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等性质需要外部 generated-property CSV 才能公平评估。
2. `plogp` 当前给 SUCC numeric token 的本地 proxy 是 LogP；正式报告需要用外部/官方 pLogP scorer。
3. direct-SMILES 现有 checkpoint 主要是 zero-source de novo 训练出来的。source-conditioned external benchmark 最终应该接一版 source-edit SFT/RL 或 agentic revise loop。
4. 当前 external group-RL 的训练 reward 只能直接优化本地 proxy subset（QED / pLogP proxy / SA when available）和 source similarity；BBBP / HIA / mutagenicity / hERG / DILI / PAMPA 等 oracle 性质需要后续 predictor-in-loop / agentic oracle loop。

## 接下来

1. ~~跑 external group-RL P1 pilot~~ → **`16774795` 完成**；proxy **40.7%**，Sim=0。
2. ~~one-shot baseline~~ → **`16779361` 完成**；proxy **42.5%**，Sim=0；≈ group-RL，RL 无增益。
3. ~~source-copy sanity~~ → **`16779362` 完成**；Sim@0.4 **100%**；排除 eval 链路 bug。
4. ~~**source-edit SFT warm-start**~~ → **`16800837` 完成**；validity **97.3%**，proxy **45.4%**，Sim 仍 **0**。
5. ~~**source-edit group-RL**~~ → **`16806903` 完成**；validity **96.4%**，proxy **45.0%**，Sim 仍 **0**；RL 无 Sim 增益。
6. ~~**agentic revise 2-step**~~ → **`16813212` 完成**；Sim@0.4 **15.6%**；validity **100%**；proxy **46.7%**。
7. ~~**agentic revise 4-step**~~ → **`16819986` 完成**；与 2-step **完全相同**（Sim **15.6%**）→ 瓶颈在 candidate pool / edit actions，不在 step depth。
8. ~~**agentic revise rich v2**~~ → **`16824486` 完成**；Sim@0.4 **32.3%**（+16.7pp vs 2-step）；validity **100%**；proxy **46.7%**。
9. ~~**GraphEditDSL agent v1**~~ → **`16825306` 完成**；Sim@0.4 **26.2%**（heuristic planner；低于 rich v2）；架构方向有效。
10. ~~**policy GraphEditDSL v2**~~ → **`16892025` 完成**；yield **2046/row**（↑ vs 89）；Sim@0.4 **19.7%**（↓ vs heuristic **26.2%**）；rich v2 仍 best **32.3%**。
11. ~~**one-shot official SR 重跑**~~ → **`16894722` 完成**；candidate-level SR **0**（无 oracle）；Sim diagnostic **0.05%**；direct 无 source-edit。
12. ~~**flight sweep**~~ → **6/6 完成**；**heuristic GraphEditDSL 2-step Sim 45.6%**（assisted best）；rich x2 **42.5%**（+10.2pp vs rich v2）；direct 训练线仍 Sim≈0。
13. **group-RL / assisted 线 official SR 重跑**；ADMET oracle CSV 回填；GraphEditDSL similarity/property 平衡或 LLM planner；C-MuMO 复跑。

## 外部来源

- GeLLMO official repo: https://github.com/ninglab/GeLLMO
- GeLLMO-C official repo: https://github.com/ninglab/GeLLMO-C
- GeLLMO ACL page: https://aclanthology.org/2025.acl-long.1225/
