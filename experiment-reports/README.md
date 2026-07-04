# 实验报告

本目录存放各实验的可更新报告（Markdown）。

## 当前实验索引

| 文件夹 | 报告文件 | 说明 |
| --- | --- | --- |
| `.` | [moleditrl-baseline-comparison.md](moleditrl-baseline-comparison.md) | MolEditRL vs SUCC v1/v2/v3 / Unified 对照 |
| `understanding-condition/` | [benchmark-roadmap-rl-agentic.md](understanding-condition/benchmark-roadmap-rl-agentic.md) | **新增** benchmark 地图 + RL/agentic 论文主线规划 |
| `unified-3m/` | [moledit-instruct-v1.md](unified-3m/moledit-instruct-v1.md) | Unified 3M MolEdit v1（`15810199`/`15837897`） |
| `unified-3m/` | [moledit-sourceanchored-v2.md](unified-3m/moledit-sourceanchored-v2.md) | **完成** source-anchored v2（`15866599`/`15884514-21`/`15884578`） |
| `understanding-condition/` | [moledit-instruct-v1.md](understanding-condition/moledit-instruct-v1.md) | SUCC MolEdit v1（`15838733`/`15838734`） |
| `understanding-condition/` | [moledit-instruct-v2-fix.md](understanding-condition/moledit-instruct-v2-fix.md) | **完成** SUCC v2 fix（`15866598`/`15866600`/`15897034`） |
| `understanding-condition/` | [moledit-instruct-v3-attack.md](understanding-condition/moledit-instruct-v3-attack.md) | **完成** SUCC v3 attack（`15901616`/`15902241`，10/10 Table1） |
| `understanding-condition/` | [moledit-instruct-dualmode-v1.md](understanding-condition/moledit-instruct-dualmode-v1.md) | dual-mode v1（`15963804`/`16006990`/`16006991`） |
| `understanding-condition/` | [moledit-instruct-dualmode-v2-guarded.md](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) | dual-mode v2 guarded（`16018414` 链；guard failed） |
| `understanding-condition/` | [moledit-instruct-dualmode-v3-guarded.md](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) | dual-mode v3 guarded（`16027758` 链；guard passed） |
| `understanding-condition/` | [moledit-instruct-dualmode-v4-warmstart-v1.md](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) | **完成** dual-mode v4 warmstart（`16042564` 链；guard pass，strict 未达目标） |
| `understanding-condition/` | [materializer-hybrid-sweep.md](understanding-condition/materializer-hybrid-sweep.md) | **完成** zero-source hybrid materializer sweep（`16056226`–`16056231`） |
| `understanding-condition/` | [materializer-random-sanity-sweep.md](understanding-condition/materializer-random-sanity-sweep.md) | **完成** random shortlist sanity sweep（`16075242`–`16075253`） |
| `understanding-condition/` | [direct-smiles-denovo-v0.md](understanding-condition/direct-smiles-denovo-v0.md) | **完成** direct SMILES de novo v0（`16079256`/`16079257`；strict≈0，mode collapse） |
| `understanding-condition/` | [direct-smiles-denovo-v1-sampled-rerank.md](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) | **完成** direct SMILES v1（best SFT n=256 **56.2%**；DPO v1 **52.1%** 未提升；RL **23.2%** collapse） |
| `understanding-condition/` | [external-multiproperty-benchmark.md](understanding-condition/external-multiproperty-benchmark.md) | **MuMO SR 48.3%**；**ADMET-prior SR 54.3%**；**C-MuMO SR 1.4%**；IND-hard **DPQ +4.5pp** |
| `understanding-condition/` | [external-paper-baselines.md](understanding-condition/external-paper-baselines.md) | **新增** 外部论文 baseline 数值目标（MolEditRL / GeLLM3O / GeLLMO-C / C-MORAL） |
| `understanding-condition/` | [paper-benchmark-plan.md](understanding-condition/paper-benchmark-plan.md) | **active** 论文 benchmark 执行计划（P0 已完成） |
| `understanding-condition/` | [direct-smiles-denovo-v2-mixed-condition.md](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) | **完成** direct SMILES v2（2p7p group RL n=256 **90.9%**；OOD conservative n=256 **89.4%**） |
| `understanding-condition/` | [denovo-2p7p-benchmark.md](understanding-condition/denovo-2p7p-benchmark.md) | **完成** de novo 2p–7p（baseline / v2_fix / dualmode） |
| `understanding-condition/` | [denovo-ood-benchmark.md](understanding-condition/denovo-ood-benchmark.md) | **完成** de novo OOD（v2_fix + dualmode） |
| `understanding-condition/` | [source-neighbor-v2-residual-ink.md](understanding-condition/source-neighbor-v2-residual-ink.md) | source-neighbor 对照（`15821981`/`15821983`） |

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v0）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16079256` | `succ-direct-smiles-2p7p` | 完成（9m51s） | [direct SMILES v0](understanding-condition/direct-smiles-denovo-v0.md) |
| `16079257` | `succ-direct-smiles-ood` | 完成（4m19s） | [direct SMILES v0](understanding-condition/direct-smiles-denovo-v0.md) |

2p7p strict **0.07%**（6000 行仅 1 个 unique SMILES）；OOD strict **0%**，validity **9.4%**。

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v1）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16079726` | `succ-direct-smiles-2p7p`（v1 sampled rerank） | 完成（1h50m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16079727` | `succ-direct-smiles-ood`（v1 sampled rerank） | 完成（23m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

v1 打破 mode collapse：2p7p strict **24.4%**（unique 5967/6000）；OOD strict **39%**（validity 95.4%）。

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v1 n=64）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16104381` | `succ-direct-smiles-2p7p-n64` | 完成（3h21m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16104383` | `succ-direct-smiles-ood-n64` | 完成（38m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

n=64 推理-only（复用 v1 ckpt）：2p7p **35.4%**（+11pp vs n=32）；OOD **54.5%**（+15.5pp）。

## 近期 Slurm Job 一览（2026-06-15 direct SMILES v1 n=128）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16143574` | `succ-direct-smiles-2p7p-n128` | 完成（6h35m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16143575` | `succ-direct-smiles-ood-n128` | 完成（1h17m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

n=128 推理-only：2p7p **45.9%**（2p **85%** > SketchMol ref）；OOD **70.9%**。

## 近期 Slurm Job 一览（2026-06-17 direct SMILES v1 n=256）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16245062` | `succ-direct-smiles-2p7p-n256` | 完成（13h45m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16245063` | `succ-direct-smiles-ood-n256` | 完成（2h35m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

n=256 推理-only（time=20h/8h）：2p7p **56.2%**（+10.3pp vs n=128）；OOD **80.1%**（+9.2pp）。

## 近期 Slurm Job 一览（2026-06-18~19 direct SMILES 并行 fast2 + RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16303776` | `succ-direct-smiles-2p7p-n256-fast` | **Timeout**（10h） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16303778` | `succ-direct-smiles-ood-n256-fast` | 完成（2h49m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16303779` | `succ-direct-smiles-2p7p-rl` | 完成（1h50m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16303783` | `succ-direct-smiles-2p7p-rl-n256` | 完成（3h07m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16338557` | `succ-direct-smiles-2p7p-n256-fast2` | 完成（15h39m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |
| `16338559` | `succ-direct-smiles-ood-n256-fast2` | 完成（2h47m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

fast2 2p7p **56.1%**（≈串行 56.2%）；OOD **78.8%**；RL n=256 strict **23.2%**（collapse，不可用）。

## 近期 Slurm Job 一览（2026-06-19~20 direct SMILES preference DPO）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16406939` | `succ-direct-smiles-dpo-2p7p` | 完成（16h08m） | [direct SMILES v1](understanding-condition/direct-smiles-denovo-v1-sampled-rerank.md) |

1 epoch preference DPO（pref n=16 + bench n=256）：2p7p strict **52.1%**（vs SFT 56.2%，**-4.1pp**）；7p **18.8%**（vs 25.8%）；无 collapse 但未学到有效 preference（`policy_pref_rate≈50%`）。

## 近期 Slurm Job 一览（2026-06-21 direct SMILES v2 mixed condition）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16472651` | `succ-direct-smiles-2p7p-v2` | **Node Fail**（epoch 9/12，17m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16472652` | `succ-direct-smiles-ood-v2` | 完成（~1h28m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16477041` | `succ-direct-smiles-2p7p-v2`（resume） | 完成（3h17m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

v2（`append_property_program` + property-count curriculum，n=64 推理）：2p7p **68.1%**（+11.9pp vs v1 n=256）；OOD **53.1%**（7p bucket **51%** vs v1 7.0%）。

## 近期 Slurm Job 一览（2026-06-21 direct SMILES v2 follow-up）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16484026` | `succ-direct-smiles-2p7p-v2-n128` | 完成（6h32m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16484027` | `succ-direct-smiles-ood-v2-n128` | 完成（1h12m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16484028` | `succ-direct-smiles-ood-v2-validity` | 完成（1h12m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16484029` | `succ-direct-smiles-ood-v2-balanced` | 完成（42m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

2p7p n=128 **79.1%**（+11pp vs n=64）；OOD n=128 **67.3%**；validity-repair **74.1%**（7p **72%**）；balanced retrain 7p 崩塌至 **23%**。

## 近期 Slurm Job 一览（2026-06-22 direct SMILES v2 main pipeline suite）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16509026` | `succ-dsm-v2-main-2p7p-default` | 完成（6h38m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16509027` | `succ-dsm-v2-main-2p7p-conservative` | 完成（6h26m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16509028` | `succ-dsm-v2-main-ood-default` | 完成（1h12m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16509029` | `succ-dsm-v2-main-ood-conservative` | 完成（1h12m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

Suite `direct_v2_main_pipeline_0622`：2p7p default **79.1%** / conservative **77.6%**；OOD default **67.3%** / conservative **74.1%**（与首轮 follow-up 可复现）。

## 近期 Slurm Job 一览（2026-06-22 direct SMILES v2 RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16532803` | `succ-direct-smiles-2p7p-v2-rl` | 完成（~4h） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

保守 REINFORCE（1 epoch，rollouts=8，sft_weight=1.0，bench n=128）：2p7p strict **37.4%**（vs SFT 79.1%）；评测未对齐 `append_property_program`（已在 `6974f20` 修复）。

## 近期 Slurm Job 一览（2026-06-22 direct SMILES v2 RL condfix）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16547785` | `succ-direct-smiles-2p7p-v2-rl-condfix` | 完成（~4h） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

conditioning 对齐后 rerun：2p7p strict **76.5%**（vs 错误 pipeline 37.4%，**+39.1pp**）；仍略低于 SFT 79.1%（-2.6pp），REINFORCE 无正向增益。

## 近期 Slurm Job 一览（2026-06-24 direct SMILES v2 group-relative RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16583941` | `succ-direct-smiles-2p7p-v2-group-rl` | 完成（~8h55m） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

group-relative RL（rollouts=16，`group_zscore`，`reference_kl_weight=0.05`，bench n=128）：2p7p strict **84.5%**（vs SFT 79.1%，**+5.4pp**）；**首次 RL 超 SFT**。

## 近期 Slurm Job 一览（2026-06-24 direct SMILES v2 group RL n=256）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16612814` | `succ-direct-smiles-2p7p-v2-group-rl-n256` | 完成（推理-only） | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

复用 group RL ckpt，`RUN_TRAIN=0`，bench n=256：2p7p strict **90.9%**（vs n=128 84.5%，**+6.4pp**）；7p **78.3%** 超 SketchMol ref；**当前 2p7p best**。

## 近期 Slurm Job 一览（2026-06-24 direct SMILES v2 OOD group-relative RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16742519` | `succ-direct-smiles-ood-v2-group-rl` | 完成 | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |

OOD group-relative RL（rollouts=16，`group_zscore`，bench n=128）：OOD overall strict **75.6%**（vs SFT 67.3%，**+8.3pp**）；7p 59.0% 弱于 conservative 主因是 default decoding（P0 follow-up 已修复）。

## 近期 Slurm Job 一览（2026-06-27 P0 paper benchmark batch）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16768165`–`16768167` | Table1 fair（eval→bench→table 链） | 完成 | [paper benchmark plan](understanding-condition/paper-benchmark-plan.md) |
| `16768168` | `succ-direct-smiles-ood-v2-group-rl-conservative` | 完成 | [direct SMILES v2](understanding-condition/direct-smiles-denovo-v2-mixed-condition.md) |
| `16768169` | `succ-direct-smiles-ood-v2-group-rl-n256` | 完成 | 同上 |
| `16768170` | `succ-direct-smiles-ood-v2-group-rl-conservative-n256` | 完成 | 同上 |

Table1 fair：mean `Acc_all(0.65)` **28.6%**（similarity rerank）。Table1 attack sweep：**n=20 → 8.5%**，**n=256 → 28.9%**，**n=2048 → 79.4%**（复现 v3 attack **0.794**）。OOD follow-up：conservative n=256 **89.4%** overall best。

## 近期 Slurm Job 一览（2026-07-03 Table1 candidate budget sweep）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `17110244` | `succ-t1-bench-n20` | 完成（~15m） | [paper benchmark plan](understanding-condition/paper-benchmark-plan.md) |
| `17110245` | `succ-t1-table-n20` | 完成 | 同上 |
| `17110247` | `succ-t1-bench-n256` | 完成（~18m） | 同上 |
| `17110248` | `succ-t1-table-n256` | 完成 | 同上 |
| `17116806` | `succ-t1-bench-n2048` | 完成（~28m） | 同上 |
| `17116807` | `succ-t1-table-n2048` | 完成 | 同上 |

`edit_latent_table_success_rerank` on `table1_benchmark_synthetic`：Acc@0.65 **8.5%**（n=20）→ **28.9%**（n=256）→ **79.4%**（n=2048，复现 v3 attack）；Acc@0.15 **61.8%** → **79.1%** → **85.4%**。GSK3B↑ 仍 **0**。

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_table1_candidate_sweep.sh
```

## 近期 Slurm Job 一览（2026-06-27 external MuMO group-RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16774795` | `succ-external-mumo-group-rl` | 完成 | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

MuMO official train/test；10 tasks × 200；group-RL warm-start 2p7p ckpt；n=20 bench。Proxy eval prop frac **40.7%**；validity **85.1%**；strict **0**（无 oracle CSV）；source Sim@0.4 **0**（需 source-edit 能力）。

## 近期 Slurm Job 一览（2026-06-27 MuMO diagnostics）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16779362` | `succ-external-mumo-source-copy` | 完成（~23s） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |
| `16779361` | `succ-external-mumo-one-shot` | 完成（~32m） | 同上 |

source-copy sanity：Sim@0.4 **100%**（eval 链路正常）。one-shot baseline：proxy **42.5%**，Sim=0；≈ group-RL **40.7%**，确认 **source-edit 能力** 是主瓶颈。

## 近期 Slurm Job 一览（2026-06-27 MuMO source-edit SFT）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16800837` | `succ-external-mumo-source-edit-sft` | 完成 | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

`append_source_property_program` + 1 epoch MuMO SFT：validity **97.3%**（+6.6pp vs one-shot）；proxy **45.4%**（+2.9pp）；Sim@0.4 仍 **0** → 待 source-edit group-RL。

## 近期 Slurm Job 一览（2026-06-27 MuMO source-edit group-RL）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16806903` | `succ-external-mumo-source-edit-rl` | 完成 | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

从 source-edit SFT ckpt 接 group-RL（`source_similarity_weight=2.0`）：validity **96.4%**；proxy **45.0%**；Sim@0.4 仍 **0** → RL 未能拉回 source conservation；待加长 SFT 或 agentic revise。

## 近期 Slurm Job 一览（2026-06-27 MuMO agentic revise）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16813212` | `succ-external-mumo-agentic-revise` | 完成（~1h48m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

2-step agentic revise（assisted line）：validity **100%**；proxy **46.7%**；Sim@0.4 **15.6%**（direct SFT/RL 仍为 0）→ 显式 local edit 方向有效。

## 近期 Slurm Job 一览（2026-06-28 MuMO GraphEditDSL agent）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16825306` | `succ-external-mumo-graph-edit` | 完成（~22m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

GraphEditDSL agent（heuristic planner）：validity **100%**；proxy **46.7%**；Sim@0.4 **26.2%**（>15.6% baseline，< rich v2 **32.3%**）→ 架构有效，待 LLM/policy planner。

## 近期 Slurm Job 一览（2026-06-28 MuMO policy GraphEditDSL v2）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16892025` | `succ-external-mumo-graph-policy` | 完成（~5h42m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

policy GraphEditDSL v2：yield **2046/row**（↑ vs heuristic 89）；Sim@0.4 **19.7%**（↓ vs heuristic **26.2%**）；rich v2 仍 best **32.3%**。

## 近期 Slurm Job 一览（2026-07-01 MuMO/C-MuMO official SR suite）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16997790` | `succ-mumo-official-copy` | 完成 | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |
| `16997792` | `succ-mumo-official-graph` | 完成 | 同上 |
| `17010444` | `succ-mumo-graph-sourceonly` | 完成 | 同上 |
| `17010445` | `succ-mumo-graph-1step` | 完成 | 同上 |
| `17010954` | `succ-cmumo-official-copy` | 完成 | 同上 |
| `17010957` | `succ-cmumo-official-graph` | 完成 | 同上 |
| `17047446` | `succ-ext-oracle` | 完成 | 同上 |

Official suite **6/6 完成** + oracle build **`17047446` 完成**。MuMO GraphEdit 2-step oracle SR **48.3%**（IND **28.1%** / OOD **69.0%**）；C-MuMO 待 source-side oracle。

## 近期 Slurm Job 一览（2026-07-02 external parallel followup）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `17050708` / `17052459` | `succ-ext-oracle-sourcefix` | 完成 | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |
| `17050919` / `17052460` | `succ-cmumo-official-copy` | 完成 | 同上 |
| `17050922` / `17052461` | `succ-cmumo-official-graph` | 完成（~6.5h） | 同上 |
| `17052462` | `succ-mumo-indhard-balanced` | 完成（~6h） | 同上 |
| `17052463` | `succ-mumo-indhard-property` | 完成（~5h） | 同上 |
| `17052464` | `succ-mumo-indhard-wide` | 完成（~7h） | 同上 |
| `17052465` | `succ-mumo-indhard-oracle` | 完成 | 同上 |
| `17052466`–`17052468` | IND-hard re-eval ×3 | 完成 | 同上 |

Parallel followup **全链完成**。IND-hard balanced：**DPQ 18.0%**（+4.5pp vs full suite）；aggregate SR **12.4%**（4 tasks）。C-MuMO source oracle 覆盖 bug 已定位（7/1776）。

## 近期 Slurm Job 一览（2026-07-02 C-MuMO dedicated oracle）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `17081083` | `succ-cmumo-oracle-v1` | 完成（~19m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |
| `17081084` | `succ-cmumo-official-copy` | 完成 | 同上 |
| `17081085` | `succ-cmumo-official-graph` | 完成（~6h） | 同上 |

C-MuMO dedicated oracle：**source_smiles 1776/1776** 覆盖。GraphEdit 2-step official SR **1.4%**（IND **1.7%** / OOD **1.1%**）；Sim≥0.4 **99.95%**。相对 GeLLMO-C（IND ~75% / OOD ~63%）差距巨大；瓶颈在 maintain/improve 多性质满足，不在 Sim。

## 近期 Slurm Job 一览（2026-07-03/04 MuMO ADMET-prior parallel）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `17138952`–`17138963` | `succ-mumo-admet-{task}` ×10 | 完成（~2.5–4h/task） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |
| `17138965` | `succ-mumo-admet-merge` | 失败（Slurm export 逗号 bug；已修脚本） | 同上 |
| `17152716` | `succ-mumo-admet-prior-oracle` | 完成（~45m） | 同上 |
| `17152717` | `succ-mumo-admet-prior-reeval` | 完成（~8m） | 同上 |

Merge 后本地重跑 + oracle/re-eval 链。**Official SR 54.3%**（IND **35.0%** / OOD **73.7%**）；top-40 vs baseline top-20 **48.3%**（+6.0pp）。

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_cmumo_dedicated_oracle_fix.sh
```

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16946786` | `succ-external-mumo-graph-policy-sim` | 完成（~5h54m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |
| `16946787` | `succ-external-mumo-graph-heur2` | 完成（~12h04m） | 同上 |
| `16946788` | `succ-external-mumo-agentic-rich-x2` | 完成（~1d5h） | 同上 |
| `16946790` | `succ-external-mumo-source-edit-sft-long` | 完成（~2h13m） | 同上 |
| `16946791` | `succ-external-mumo-source-edit-rl-official` | 完成（~1h38m） | 同上 |
| `16946792` | `succ-external-mumo-source-edit-rl-highsim` | 完成（~3h50m） | 同上 |

rich x2（4096/row）：Sim@0.4 **42.5%**（+10.2pp vs rich v2 **32.3%**）；仍略低于 heuristic GraphEdit 2-step **45.6%**。

**heuristic GraphEditDSL 2-step Sim 45.6%**（assisted 新 best）；official top-20 Sim **46%** / DPQ SR **13.5%**（proxy-only）；sim-anchor = policy v2 **19.7%**；direct SFT long / high-sim RL 仍 Sim≈0。

一键提交：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_external_mumo_flight_sweep.sh
```

## 近期 Slurm Job 一览（2026-06-29 MuMO official SR 重跑）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16894722` | `succ-external-mumo-one-shot` | 完成（~31m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

`13f3bec` candidate-level official SR 重跑：validity **90.7%**；SR **0**（无 oracle）；Sim diagnostic **0.05%**；direct 确认无 source-edit。

## 近期 Slurm Job 一览（2026-06-27 MuMO agentic revise 4-step）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16819986` | `succ-external-mumo-agentic-revise-4step` | 完成（~1h23m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

4-step（复用 v1 proposals）：与 2-step **完全相同**（Sim **15.6%**，proxy **46.7%**）→ 瓶颈不在 step depth，待扩大 candidate pool / edit actions。

## 近期 Slurm Job 一览（2026-06-28 MuMO agentic revise rich v2）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16824486` | `succ-external-mumo-agentic-rich` | 完成（~10h27m） | [external multiproperty](understanding-condition/external-multiproperty-benchmark.md) |

rich v2：beam **128**，2048 candidates/row，rich edit actions + similarity-first → Sim@0.4 **32.3%**（+16.7pp vs 2-step）；proxy **46.7%** 不变。

## 近期 Slurm Job 一览（2026-06-15 hybrid 默认验证）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16072912` | `succ-denovo-2p7p-ours`（standalone hybrid 默认） | 完成（18m49s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `16072913` | `succ-denovo-ood-ours`（standalone hybrid 默认） | 完成（12m42s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |

与 materializer sweep `16056226`/`16056227` 数值一致，确认默认切换可复现。

## 近期 Slurm Job 一览（2026-06-15 random shortlist sanity）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16075242`–`16075253` | `succ-{2p7p,ood}-san-v1_k{64..4096}` | 完成 | [random sanity](understanding-condition/materializer-random-sanity-sweep.md) |

random@k=4096 已达 2p7p **97.8%** / OOD **98.0%**，与 latent hybrid 几乎重合；小 k 下 random 反而更高。

## 近期 Slurm Job 一览（2026-06-14 materializer sweep）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16056226` | `succ-2p7p-mat-v1` | 完成（1h26m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056227` | `succ-ood-mat-v1` | 完成（25m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056228` | `succ-2p7p-mat-v3` | 完成（1h27m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056229` | `succ-ood-mat-v3` | 完成（25m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056230` | `succ-2p7p-mat-v4` | 完成（1h26m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |
| `16056231` | `succ-ood-mat-v4` | 完成（24m） | [materializer sweep](understanding-condition/materializer-hybrid-sweep.md) |

## 近期 Slurm Job 一览（2026-06-14 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16042564` | `succ-univideo-mol`（dualmode v4 warmstart） | 完成（52m00s） | [dualmode v4](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) |
| `16043202`–`16043204` | Table1 eval/bench/table | 完成 | [dualmode v4](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) |
| `16043205` | `succ-table1-guard` | **Passed** | [dualmode v4](understanding-condition/moledit-instruct-dualmode-v4-warmstart-v1.md) |
| `16043206` | `succ-denovo-ood-ours`（v4 ckpt） | 完成（12m06s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16043207` | `succ-denovo-2p7p-ours`（v4 ckpt） | 完成（13m48s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-14）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16027758` | `succ-univideo-mol`（dualmode v3 训练） | 完成（54m08s） | [dualmode v3](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) |
| `16028792`–`16028794` | Table1 eval/bench/table | 完成 | [dualmode v3](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) |
| `16028795` | `succ-table1-guard` | **Passed**（Acc 0.794 ≥ 0.794） | [dualmode v3](understanding-condition/moledit-instruct-dualmode-v3-guarded.md) |
| `16028796` | `succ-denovo-ood-ours`（v3 ckpt） | 完成（12m52s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16028797` | `succ-denovo-2p7p-ours`（v3 ckpt） | 完成（13m51s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-13 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16018414` | `succ-univideo-mol`（dualmode v2 训练） | 完成（53m14s） | [dualmode v2](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) |
| `16019240`–`16019242` | Table1 eval/bench/table | 完成 | [dualmode v2](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) |
| `16019243` | `succ-table1-guard` | **Failed**（Acc 0.794 < 0.894） | [dualmode v2](understanding-condition/moledit-instruct-dualmode-v2-guarded.md) |
| `16019244` | `succ-denovo-ood-ours`（v2 ckpt） | 完成（15m05s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16019245` | `succ-denovo-2p7p-ours`（v2 ckpt） | 完成（14m20s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-13）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `16006990` | `succ-denovo-ood-dualmode` | 完成（16m09s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |
| `16006991` | `succ-denovo-2p7p-dualmode` | 完成（17m55s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |

## 近期 Slurm Job 一览（2026-06-12）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15963804` | `succ-dualmode-v1`（MolEdit+de novo+OOD 训练） | 完成（54m29s） | [dualmode v1](understanding-condition/moledit-instruct-dualmode-v1.md) |

## 近期 Slurm Job 一览（2026-06-11）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15958481` | `succ-denovo-2p7p`（baseline） | 完成（1h08m） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `15959362` | `succ-denovo-2p7p-ours` | 完成（13m46s） | [denovo 2p7p](understanding-condition/denovo-2p7p-benchmark.md) |
| `15959891` | `succ-denovo-ood-ours` | 完成（11m51s） | [denovo OOD](understanding-condition/denovo-ood-benchmark.md) |

## 历史 Slurm Job（2026-06-10）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15901616`–`15901618` | v3 attack 训练/bench/table | 完成 | [v3 attack](understanding-condition/moledit-instruct-v3-attack.md) |
| `15902238`–`15902241` | v3 Table1 扩展 eval/bench/table | 完成 | [v3 attack](understanding-condition/moledit-instruct-v3-attack.md) |
| `15898378`–`15898540` | v2 Table1 扩展 | 完成 | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |

## 历史 Slurm Job（2026-06-09 晚）

| Job ID | 名称 | 状态 | 报告 |
| --- | --- | --- | --- |
| `15866598` | `succ-univideo-mol`（v2 fix） | 完成（27m16s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866600` | `succ-univideo-bench`（v2 fix） | 完成（2m37s） | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866601` | `succ-moledit-table`（v2 fix，缺 TDC） | 完成 | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15897034` | `succ-moledit-table`（v2 fix，PyTDC） | 完成 | [v2 fix](understanding-condition/moledit-instruct-v2-fix.md) |
| `15866599` | `smu3m-srcanch-v2` | 完成（3m17s） | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |
| `15884514`–`15884521` | `smu3m-diff-bench`（5 shards） | 完成 | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |
| `15884578` | `smu3m-moledit-table` | 完成 | [source-anchored v2](unified-3m/moledit-sourceanchored-v2.md) |

新增实验时：在对应子文件夹下新建 `{实验名}.md`，并在此表加一行链接。
