# External Multi-property Benchmark Port

| 字段 | 值 |
| --- | --- |
| **状态** | MuMO **ADMET-prior v1 top-40 SR 54.3%**（主结果）；v2 `admet_hybrid` **12.5%**（失败 ablation）；baseline **48.3%**；C-MuMO **1.4%**；IND-hard **DPQ +4.5pp** |
| **最后更新** | 2026-07-04（MuMO ADMET-prior v2 jobs `17164562`–`17164574`；fair-budget de novo `17162873`–`17162880`） |
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

### Official suite 结果汇总

#### Proxy-only（缺 ADMET，lower-bound diagnostic）

**口径**：candidate-level top-20；无 ADMET CSV 时仅 **DPQ** 可标 `official`；aggregate SR **不可**与 GeLLMO 横比。

| Job | 变体 | SR (all) | Sim≥0.4 (all) | DPQ SR |
| --- | --- | ---: | ---: | ---: |
| `16997792` | 2-step + direct proposal | 1.35% | 46.0% | 13.5% |
| `17010445` | 1-step top-20 | 0.95% | 99.9% | 9.5% |
| `17010444` | source-only 2-step | 0% | 99.4% | 0% |

#### ADMET oracle-backed（`17047446` + re-eval，`generated_properties.csv`）

Oracle build：**38,377** unique SMILES → ADMET-AI → **39,522** rows merged；ADMET 覆盖 **97.1%**（1145 SMILES 缺 ADMET，多为 invalid/duplicate edge cases）。Re-eval 输出：`outputs/*/benchmark_with_oracle_v1/`（不纳入 git）。

| Job | 变体 | SR (all) | Sim(success) | IND avg SR | OOD avg SR | Internal strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **`16997792`** | **2-step + proposal** | **48.3%** | **0.19** | **28.1%** | **69.0%** | 8.4% |
| `17010444` | source-only 2-step | 8.1% | 0.51 | — | — | 7.4% |
| `17010445` | 1-step top-20 | 3.0% | 0.49 | — | — | 2.2% |
| `17010957` | C-MuMO 2-step | **0%** | — | — | — | 0% |
| **`17081085`** | **C-MuMO 2-step + dedicated oracle** | **1.4%** | **0.63** | **1.7%** | **1.1%** | **1.3%** |

MuMO 2-step task-level SR（oracle-backed，top-20）：

| Split | BDP | BDPQ | BDQ | BPQ | DPQ | BDMQ | BHMQ | BMPQ | HMPQ | MPQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SR | 5.5% | 13.5% | 20% | **88%** | 13.5% | 12.5% | 57% | 89.5% | 94.3% | 91.5% |

**结论（oracle-backed）**：

1. **2-step + direct proposal 是主线**：aggregate SR **48.3%**；OOD **69%** 接近 GeLLMO 论文量级（~77–89%），IND **28%** 仍偏低。
2. **2-step vs 1-step**：48.3% vs 3.0%——beam + proposal 在 full oracle 下增益远大于 proxy-only 诊断。
3. **source-only 仍弱于 full pipeline**（8.1% vs 48.3%），但不再 Sim≈100%/SR=0 的假象。
4. **C-MuMO SR=0**：maintain/improve 需要 **source-side** property oracle；`17047446` CSV 主要覆盖 generated SMILES，source 性质回填不完整（eval prop frac **~31%**）。
5. **2026-07-02 修复**：`collect_external_multiproperty_oracle_smiles.py` 之前每行只收第一个 SMILES，导致有 `generated_smiles` 时跳过 `source_smiles/target_smiles`。现已改为同一行收全列，并让 evaluator 在未传 `source_properties_csv` 时默认复用 generated/source unified oracle CSV。
6. **RI(success)** 修复：source property 接近 0 时 RI 记为 undefined，避免 `1e-8` denominator 造成虚高。主表仍优先报 **SR + Sim(success)**，RI 作为 supplement。

GeLLMO 参考（Table 2 best generalist）：IND SR **~77%**，OOD SR **~89%**。

## Parallel followup 结果（`0574bcf`，jobs `17050708`–`17052468`）

入口：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_parallel_followup.sh
```

两条 Slurm DAG 均已 **完成**（exit 0）：

| 链 | Jobs | 说明 |
| --- | --- | --- |
| **A：C-MuMO** | `17050708`/`17052459` → `17050919`/`17052460` → `17050922`/`17052461` | sourcefix oracle → copy sanity → GraphEdit 2-step rerun |
| **B：MuMO IND-hard** | `17052462`–`17052464` → `17052465` → `17052466`–`17052468` | BDP/BDPQ/DPQ/BDMQ 三方向 sweep → oracle → re-eval |

### 链 A：sourcefix oracle + C-MuMO official rerun

| Job | 名称 | 耗时 | 状态 |
| --- | --- | --- | --- |
| `17050708` / `17052459` | `succ-ext-oracle-sourcefix` | ~3m | **完成** |
| `17050919` / `17052460` | `succ-cmumo-official-copy` | ~1.5m | **完成** |
| `17050922` / `17052461` | `succ-cmumo-official-graph` | ~6.5h | **完成** |

sourcefix oracle：`outputs/external_oracle_build_sourcefix_v1/generated_properties.csv`（**39,522** SMILES，CSV 内 ADMET 列 **100%** 覆盖）。但 **C-MuMO benchmark source SMILES 与 oracle 键重叠仅 7/1776**（canonical 后仍如此）——oracle 增量主要来自 MuMO GraphEdit 候选池，**未覆盖 C-MuMO `test.json` 的 source 分子**。

C-MuMO GraphEdit 2-step（`17050922`/`17052461`，candidate-level top-20）：

| 指标 | 值 | 说明 |
| --- | ---: | --- |
| Valid | **100%** | 40k candidate rows |
| Sim≥0.4（candidate pool diagnostic） | **99.95%** | source preservation 极好 |
| Sim≥0.4（selected-1 diagnostic） | **98.4%** | selected-1 仍高 Sim |
| **SR (all)** | **0** | `lower_bound_missing_oracle`；eval prop frac **~31.5%** |
| Internal strict | **0** | — |

即使用 `sourcefix` CSV 直接 re-eval（`benchmark_with_sourcefix_oracle_v1/`，本地补跑），SR 仍为 **0**，eval prop frac **~31.5%**——瓶颈是 **oracle 键空间**，不是 GraphEdit 生成质量。

**结论（C-MuMO）**：

1. GraphEdit 候选 **Sim 接近满分**，但 **不能 claim official SR**。
2. 下一步需单独建 `external_oracle_build_cmumo_v1/`：输入必须含 `external_cmumo_official_copy_sanity_v1` 的 **source_smiles** + GraphEdit 候选 CSV；official suite 应设 `SUCC_OFFICIAL_BUILD_ORACLE_CSV=0` 并直接传预建 oracle，避免 `build_oracle_csv=1` 用 TDC-only 局部表覆盖 full ADMET CSV。
3. 2026-07-02 修复方向：新增 dedicated C-MuMO oracle pipeline，会从 raw C-MuMO `test.json` 重新导出 source rows、合并旧 copy/GraphEdit candidates 建 oracle，并输出 `oracle_coverage_audit.md`；evaluator 也改为 **source 缺失 / Tanimoto NaN 不再算 Sim 成功**，防止再次出现 false-positive source-preservation 诊断。

### 链 B：MuMO IND-hard GraphEdit sweep

目标 tasks：**BDP, BDPQ, DPQ, BDMQ**（MuMO 低分 IND/OOD 子集）。Oracle：`external_oracle_build_mumo_indhard_v1/`（**16,134** SMILES，ADMET **100%**）。

| Job | 变体 | selection | beam / row cap | aggregate SR | Sim(success) | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| `17052462` | **balanced** | similarity_first | 128 / 4096 | **12.4%** | **0.174** | **official** |
| `17052463` | property_first | score | 160 / 4096 | 9.5% | 0.156 | lower_bound（eval prop frac ~72%） |
| `17052464` | wide_balanced | similarity_first | 192 / 8192 | 10.0% | 0.164 | lower_bound（eval prop frac ~75%） |

**balanced** task-level SR（vs full-suite `16997792` + oracle v1）：

| Task | Split | Full suite SR | IND-hard balanced | Δ |
| --- | --- | ---: | ---: | ---: |
| BDP | ind | 5.5% | 5.5% | 0 |
| BDPQ | ind | 13.5% | 13.5% | 0 |
| DPQ | ind | 13.5% | **18.0%** | **+4.5pp** |
| BDMQ | ood | 12.5% | 12.5% | 0 |

**结论（IND-hard）**：

1. **balanced** 是唯一可 claim official 的变体；**DPQ +4.5pp** 是唯一实质增益。
2. `property_first` / `wide_balanced` 因候选 SMILES 与 merged oracle 对齐不足，re-eval 为 lower_bound；不宜与 balanced 横比 SR。
3. BDP/BDPQ/BDMQ 在 dedicated sweep 下 **未超过** full-suite 基线——MuMO IND 短板仍在多性质联合满足，而非单纯 search budget。

Re-eval 产物：`outputs/external_mumo_indhard_graph_edit_*/benchmark_with_indhard_oracle_v1/`（不进 git）。

## C-MuMO dedicated oracle + official SR（`b769919`，jobs `17081083`–`17081085`）

入口：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_cmumo_dedicated_oracle_fix.sh
```

| Job | 名称 | 耗时 | 状态 |
| --- | --- | --- | --- |
| `17081083` | `succ-cmumo-oracle-v1` | ~19m | **完成** |
| `17081084` | `succ-cmumo-official-copy` | ~1m | **完成** |
| `17081085` | `succ-cmumo-official-graph` | ~6h | **完成** |

Dedicated oracle：`outputs/external_oracle_build_cmumo_v1/generated_properties.csv`（**38,247** SMILES，ADMET **100%**）。`oracle_coverage_audit.md` 确认 **source_smiles 1776/1776**、generated 候选 **100%** 覆盖（修复前仅 **7/1776**）。

### Copy sanity（oracle 上界）

| 变体 | SR (all) | Eval prop frac | Status |
| --- | ---: | ---: | --- |
| Source-copy | **0.05%** | **100%** | official |
| Target-copy | **0.05%** | **100%** | official |

复制 source/target 几乎不可能满足 C-MuMO 的 improve+maintain 目标；说明 **oracle 链路已正常**，低 SR 来自编辑能力而非评测 bug。

### GraphEdit 2-step official SR（candidate-level top-20）

输出：`external_cmumo_official_graph_edit_heuristic_2step_cmumo_oracle_v1/`  
Re-eval：`benchmark_with_cmumo_oracle_v1/`（GraphEdit job 未自动写 report，本地补跑 re-eval）

| 指标 | 值 |
| --- | ---: |
| **SR (all)** | **1.4%** |
| **IND avg SR** | **1.7%** |
| **OOD avg SR** | **1.1%** |
| Sim(success) | **0.63** |
| Sim≥0.4（diagnostic） | **99.95%** |
| Internal strict | **1.3%** |
| Valid | **100%** |
| mean_source_tanimoto | **0.73** |

C-MuMO task-level SR（oracle-backed，top-20）：

| Split | Task | SR | Sim(success) |
| --- | --- | ---: | ---: |
| ind | ACEP | 2.0% | 0.84 |
| ind | **BPQ** | **5.0%** | 0.52 |
| ind | ELQ | 1.0% | 0.61 |
| ind | DHMQ | 0.5% | 0.55 |
| ind | BDPQ | 0% | — |
| ood | **ABMP** | **4.0%** | 0.69 |
| ood | BCMQ | 1.0% | 0.62 |
| ood | HLMPQ | 0.5% | 0.45 |
| ood | BDEQ | 0% | — |
| ood | CDE | 0% | — |

**结论（C-MuMO official SR）**：

1. **首次可 claim official SR**：aggregate **1.4%**；eval prop frac **100%**。
2. **Sim 不是瓶颈**（99.95% diagnostic），**maintain/improve 多性质联合满足** 才是主因；与 MuMO **48.3%** 差距巨大。
3. 相对 GeLLMO-C 论文 best generalist（IND **~74.8%** / OOD **~63.0%**），当前 heuristic GraphEditDSL **远未达标**。
4. 下一步：C-MuMO 专用 planner（property-specific maintain/improve objective）、或接 GeLLMO-C 风格 LLM + 我们的 evaluator 横比。

## Aligned MuMO evaluation diagnostics（`1ba9237`）

新增评测对齐层（**不改模型**）：任意 candidate CSV 可用同一套 official evaluator 重评，输出 aligned comparison 表 + failure breakdown，便于 GeLLMO official 跑完后并排比较。

```bash
SUCC_EXTERNAL_ALIGNED_LABEL=ours_graph_edit \
SUCC_EXTERNAL_ALIGNED_PREDICTION_CSV=SketchMol-Understanding-Condition/outputs/external_mumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv \
SUCC_EXTERNAL_ALIGNED_GENERATED_PROPERTIES_CSV=SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv \
SUCC_EXTERNAL_ALIGNED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/external_aligned_ours_graph_edit_v1 \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_aligned_eval.sh
```

GeLLMO postprocess 完成后合并：

```bash
python SketchMol-Understanding-Condition/scripts/summarize_external_multiproperty_aligned_runs.py \
  --run ours=.../external_aligned_ours_graph_edit_v1/eval_ours_graph_edit \
  --run gellmo=.../external_gellmo_official_mumo_v1/eval_official_oracle \
  --output-dir SketchMol-Understanding-Condition/outputs/external_aligned_ours_vs_gellmo_v1
```

### MuMO method repair：ADMET-prior GraphEditDSL ✅（jobs `17138952`–`17152717`）

目标：提升 **我们的方法在 MuMO benchmark 上的 SR**，尤其是 IND / IND-hard。当前 GraphEditDSL 生成和排序主要依赖 source similarity + 本地 proxy（QED / pLogP / SA），对 BBBP / DRD2 / HIA / mutagenicity 这类 ADMET oracle 性质基本是盲生成。新线新增 `admet_prior_graph_dsl`：

- 不读取 official oracle 做 test-time selection；official oracle 仍只用于最终 re-eval。
- 在 graph-edit planner 中加入可解释 ADMET templates：BBB/DRD2 倾向疏水芳香/弱碱胺、HIA 控制极性和柔性、mutagenicity↓ 倾向删除/替换潜在 reactive handles。
- 在 candidate ranking 中加入 `admet_prior_score`，由 RDKit descriptors / simple structural alerts 计算，作为 beam expansion 和 top-k 的非 oracle 先验。
- 一键入口：`submit_external_mumo_admet_prior_repair.sh`（默认 **10 task 并行** + checkpoint + merge）；DAG 为 `GraphEdit ADMET-prior ×10 -> merge -> oracle -> re-eval`。

```bash
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_admet_prior_repair.sh
# 单 job 回退：SUCC_MUMO_ADMET_PARALLEL_TASKS=0 bash ...
# 仅重跑 merge+oracle（shard 已完成）：
bash SketchMol-Understanding-Condition/scripts/run_external_mumo_admet_prior_merge_task_shards.sh
```

**基础设施（2026-07-03/04）**：

- 10 个 task 并行 Slurm job（各 ~200 rows，~2.5–4h）；`build_external_graph_edit_agent_predictions.py` 增量 checkpoint（每 10 行）。
- merge 脚本 `merge_external_graph_edit_task_shards.py`；plan jsonl 流式拼接（~24GB 总量）。
- merge job `17138965` 曾失败：Slurm `--export` 把 `BDP,BDQ,...` 截断为 `BDP`；已改为 `parallel_task_list.txt`。

**Official SR（`17152717`，top-40 candidates，oracle `17152716`）**：

| Task | Split | SR | Sim(success) | 对比 baseline 2-step top-20 |
| --- | --- | ---: | ---: | --- |
| BDP | ind | 6.0% | 0.51 | baseline aggregate **48.3%** |
| BDPQ | ind | 21.0% | 0.15 | IND **28.1%** |
| BDQ | ind | 32.5% | 0.17 | OOD **69.0%** |
| BPQ | ind | **93.0%** | 0.21 | |
| DPQ | ind | 22.5% | 0.14 | |
| BDMQ | ood | 26.0% | 0.16 | |
| BHMQ | ood | 63.0% | 0.22 | |
| BMPQ | ood | **91.5%** | 0.19 | |
| HMPQ | ood | **94.8%** | 0.14 | |
| MPQ | ood | **94.0%** | 0.26 | |
| **all** | | **54.3%** | **0.20** | **+6.0pp** vs **48.3%** |

Aggregate 分解：IND **35.0%**（+6.9pp）；OOD **73.7%**（+4.7pp）。

⚠️ **口径**：ADMET-prior 用 **top-40**；baseline `16997792` 用 **top-20**；beam/search 空间也更大（beam 192 vs 128）。对外 claim 须标注 candidate budget + planner config。

产物：

- Shards：`outputs/external_mumo_graph_edit_admet_prior_v1/tasks/{BDP,...}/`
- Merged：`outputs/external_mumo_graph_edit_admet_prior_v1/benchmark_graph_edit_agent/`
- Official：`outputs/external_mumo_graph_edit_admet_prior_v1/benchmark_with_admet_prior_oracle_v1/`

推荐先跑 full MuMO；若想快筛 IND-hard，可设置 `SUCC_MUMO_ADMET_TASKS=BDP,BDPQ,DPQ,BDMQ`。

### MuMO ADMET-prior v2：`admet_hybrid` ranking ❌（jobs `17164562`–`17164574`）

v1 的 `similarity_first` 用 `local_success_fraction` 排序，但 local proxy 只覆盖 plogp/qed/sa。v2 在 Sim≥0.4 桶内**优先按 `admet_prior_score` 排序**（`admet_hybrid`），超参对齐 IND-hard balanced sweep。

**Official SR（`17164574`，top-40）**：

| Task | Split | v1 SR | v2 SR | Δ |
| --- | --- | ---: | ---: | ---: |
| BDP | ind | 6.0% | 4.5% | −1.5pp |
| BDPQ | ind | 21.0% | 0.5% | −20.5pp |
| BDQ | ind | 32.5% | 0.5% | −32.0pp |
| BPQ | ind | 93.0% | 34.0% | −59.0pp |
| DPQ | ind | 22.5% | 0.5% | −22.0pp |
| BDMQ | ood | 26.0% | 0% | −26.0pp |
| BHMQ | ood | 63.0% | 11.0% | −52.0pp |
| BMPQ | ood | 91.5% | 22.5% | −69.0pp |
| HMPQ | ood | 94.8% | 12.0% | −82.8pp |
| MPQ | ood | 94.0% | 40.0% | −54.0pp |
| **all** | | **54.3%** | **12.5%** | **−41.8pp** |

Aggregate 分解：IND **~8%**；OOD **~17%**。Sim(success) **0.52**（v1 **0.20**）——prior 选出高相似、descriptor 看似合理但 **过不了 oracle** 的候选。

**结论**：v2 是 negative ablation；MuMO 主结果仍用 v1 **54.3%**。入口保留：`submit_external_mumo_admet_prior_v2_repair.sh`；产物：`outputs/external_mumo_graph_edit_admet_prior_v2/benchmark_with_admet_prior_oracle_v1/`。

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

source-oracle 修复后推荐重跑（增量补 ADMET；旧 CSV 已覆盖的 SMILES 会跳过）：

```bash
OLD_ORACLE=SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv
NEW_ORACLE=SketchMol-Understanding-Condition/outputs/external_oracle_build_sourcefix_v1/generated_properties.csv
SUCC_ORACLE_SLURM_JOB_NAME=succ-ext-oracle-sourcefix \
SUCC_ORACLE_WORK_DIR=SketchMol-Understanding-Condition/outputs/external_oracle_build_sourcefix_v1 \
SUCC_ORACLE_OUTPUT_CSV=$NEW_ORACLE \
SUCC_ORACLE_MERGE_PROPERTIES_CSV=$OLD_ORACLE \
SUCC_ORACLE_INPUT_CSV=SketchMol-Understanding-Condition/outputs/external_mumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/external_multiproperty_rows.csv,SketchMol-Understanding-Condition/outputs/external_cmumo_official_graph_edit_heuristic_2step_v1/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_generated_oracle_pipeline.sh

SUCC_OFFICIAL_RUN_MUMO=0 \
SUCC_OFFICIAL_RUN_CMUMO=1 \
SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV=$NEW_ORACLE \
SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV=$NEW_ORACLE \
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_official_suite.sh
```

如果要把可并行工作一次性提交，用：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_multiproperty_parallel_followup.sh
```

这个入口会同时做两条事：

1. 提交 `succ-ext-oracle-sourcefix`，并把 C-MuMO official rerun 通过 `afterok:<oracle_job>` 挂在它后面。
2. 立刻提交 MuMO IND-hard sweep（BDP / BDPQ / DPQ / BDMQ），再把这些新候选接到独立的 ADMET oracle + re-eval 链上。

依赖关系：C-MuMO official 必须等 sourcefix oracle；MuMO IND-hard sweep 的 candidate generation 可以同步跑，不需要等 C-MuMO。

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
13. ~~**parallel followup**~~ → **完成**（`17050708`–`17052468`）：IND-hard balanced **DPQ +4.5pp**；C-MuMO source oracle 覆盖 bug 已定位。
14. ~~**C-MuMO dedicated oracle**~~ → **完成**（`17081083`–`17081085`）：official SR **1.4%**（IND **1.7%** / OOD **1.1%**）；Sim≥0.4 **99.95%**。
15. ~~**MuMO method repair**~~ → **完成**（`17138952`–`17152717`）：ADMET-prior top-40 SR **54.3%**（IND **35.0%** / OOD **73.7%**）；+6.0pp vs baseline **48.3%**（⚠️ top-40 vs top-20）。
16. ~~**MuMO ADMET-prior v2 `admet_hybrid`**~~ → **完成**（`17164562`–`17164574`）：SR **12.5%**（−41.8pp vs v1）；negative ablation；主结果仍 v1。
17. ~~**De novo fair-budget suite**~~ → **完成**（`17162873`–`17162880`）：2p-7p @40 **68.1%**；OOD @40 **48.1%**。
18. **GeLLMO official baseline**（`17081866`–`17081876` pending GPU）；完成后用 aligned eval 与 ours 并排。
19. **MolEdit Table1 direct group RL**（`17161387` pending GPU）。
20. C-MuMO 专用 maintain/improve planner；GraphEditDSL similarity/property 平衡或 LLM planner。

## 外部来源

- GeLLMO official repo: https://github.com/ninglab/GeLLMO
- GeLLMO-C official repo: https://github.com/ninglab/GeLLMO-C
- GeLLMO ACL page: https://aclanthology.org/2025.acl-long.1225/
