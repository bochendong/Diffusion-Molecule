# SUCC Direct SMILES De Novo v2 Mixed Condition

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-21） |
| **最后更新** | 2026-06-27（P0 OOD follow-up `16768168`–`16768170`；Table1 fair `16768165`–`16768167`） |
| **入口** | `submit_direct_smiles_denovo_2p7p_v2_benchmark.sh` / `submit_direct_smiles_denovo_ood_v2_benchmark.sh` / `submit_direct_smiles_denovo_v2_rerun_suite.sh` / `submit_direct_smiles_rl_2p7p_v2.sh` / `submit_direct_smiles_group_rl_2p7p_v2.sh` / `submit_direct_smiles_group_rl_ood_v2.sh` |
| **目的** | 在 direct SMILES decoder 上引入 mixed conditioning + property-count curriculum，提升高 property count（6p/7p）strict |

## Slurm 运行记录

| Job | 名称 | 墙钟 | Exit | 输出 |
| --- | --- | --- | --- | --- |
| `16472651` | `succ-direct-smiles-2p7p-v2`（首轮；epoch 9/12 时 Node Fail） | 17m | **Node Fail** | `direct_smiles_denovo_2p7p_v2_mixed_condition/`（`latest_checkpoint.pt` epoch 9） |
| `16472652` | `succ-direct-smiles-ood-v2` | ~1h28m | 0 | `direct_smiles_denovo_ood_v2_mixed_condition/` |
| `16477041` | `succ-direct-smiles-2p7p-v2`（从 epoch 9 resume） | 3h17m | 0 | 同上（补完 epoch 10–12 + n=64 评测） |
| `16484026` | `succ-direct-smiles-2p7p-v2-n128`（推理-only，复用 v2 ckpt） | 6h32m | 0 | `direct_smiles_denovo_2p7p_v2_mixed_condition_n128/` |
| `16484027` | `succ-direct-smiles-ood-v2-n128`（推理-only，复用 v2 ckpt） | 1h12m | 0 | `direct_smiles_denovo_ood_v2_mixed_condition_n128/` |
| `16484028` | `succ-direct-smiles-ood-v2-validity`（保守 decoding，不重训） | 1h12m | 0 | `direct_smiles_denovo_ood_v2_mixed_condition_validity_repair_n128/` |
| `16484029` | `succ-direct-smiles-ood-v2-balanced`（curriculum 只开 sampling） | 42m | 0 | `direct_smiles_denovo_ood_v2_mixed_condition_balanced/` |
| `16509026` | `succ-dsm-v2-main-2p7p-default`（suite `direct_v2_main_pipeline_0622`） | 6h38m | 0 | `.../direct_v2_main_pipeline_0622/2p7p_default_n128/` |
| `16509027` | `succ-dsm-v2-main-2p7p-conservative` | 6h26m | 0 | `.../2p7p_conservative_n128/` |
| `16509028` | `succ-dsm-v2-main-ood-default` | 1h12m | 0 | `.../ood_default_n128/` |
| `16509029` | `succ-dsm-v2-main-ood-conservative` | 1h12m | 0 | `.../ood_conservative_n128/` |
| `16532803` | `succ-direct-smiles-2p7p-v2-rl`（保守 REINFORCE + bench n=128） | ~4h | 0 | `direct_smiles_denovo_2p7p_v2_rl_v1/` |
| `16547785` | `succ-direct-smiles-2p7p-v2-rl-condfix`（conditioning fix 后 rerun） | ~4h | 0 | `direct_smiles_denovo_2p7p_v2_rl_v2_conditionfix/` |
| `16583941` | `succ-direct-smiles-2p7p-v2-group-rl`（group-relative RL + bench n=128） | ~8h55m | 0 | `direct_smiles_denovo_2p7p_v2_group_rl_v1/` |
| `16612814` | `succ-direct-smiles-2p7p-v2-group-rl-n256`（group RL ckpt，推理-only n=256） | benchmark-only | 0 | `.../benchmark_direct_smiles_group_rl_n256/` |
| `16742519` | `succ-direct-smiles-ood-v2-group-rl`（OOD group-relative RL + bench n=128） | ~3h05m | 0 | `direct_smiles_denovo_ood_v2_group_rl_v1/` |
| `16768168` | `succ-direct-smiles-ood-v2-group-rl-conservative`（group RL ckpt + conservative n=128） | benchmark-only | 0 | `.../benchmark_direct_smiles_group_rl_conservative_n128/` |
| `16768169` | `succ-direct-smiles-ood-v2-group-rl-n256`（group RL ckpt + default n=256） | benchmark-only | 0 | `.../benchmark_direct_smiles_group_rl_n256/` |
| `16768170` | `succ-direct-smiles-ood-v2-group-rl-conservative-n256`（group RL ckpt + conservative n=256） | benchmark-only | 0 | `.../benchmark_direct_smiles_group_rl_conservative_n256/` |

## v2 方法（相对 v1）

| 变量 | v1 | v2 |
| --- | --- | --- |
| condition mixing | `features_only` | **`append_property_program`** |
| property-count curriculum | 关 | **采样 + loss 加权均开**（power=1.0，baseline=2.0） |
| 推理 num_samples | 32（默认）/ 64–256 缩放 | **64**（首轮） |
| 训练 | 12 epoch SFT | 12 epoch SFT（同 lr/batch） |

仍无 molecule DB retrieval；推理仍为 sampled rerank（temperature 0.85、top-k 40、top-p 0.95 等，与 v1 一致）。

## 2p7p 结果（n=64，job `16477041`，resume 自 `16472651`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid | mean valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.924** | **0.830** | **0.747** | **0.611** | **0.523** | **0.449** | **0.681** | **0.986** | **5914 / 6000** | **18.7** |

训练末 epoch（12）：`eval_loss` **0.806**，`eval_perplexity` **2.24**；`mean_property_count` **5.16**（curriculum 生效）。

### vs v1（同 n=64 或 best n=256）

| 对比 | overall strict | 6p strict | 7p strict |
| --- | ---: | ---: | ---: |
| v1 n=64（推理-only） | 0.354 | 0.128 | 0.084 |
| v1 n=256（推理-only） | 0.562 | 0.314 | 0.258 |
| **v2 n=64（重训）** | **0.681** | **0.523** | **0.449** |

v2 在 **仅 n=64 推理** 下 overall **+11.9pp** vs v1 best n=256；7p **+19.1pp**（0.258→0.449）。2p/3p/4p 亦全面超过 SketchMol ref。

## SketchMol fair-budget 对齐（待跑）

SketchMol 官方 sampling 脚本的 sample 数为 `n_samples * conditional_count`；官方 README paper-style 示例与本仓库 `run_paper_repro.sh` 默认是 `n_samples=1`、`conditional_count=40`，即 **SketchMol@40**。因此 direct-SMILES de novo 主文公平表需要补：

- **Ours@1**：single-sample / no best-of-K scaling。
- **Ours@40**：与 SketchMol paper-style candidate budget 对齐。
- **Ours@256**：保留为 high-budget/scaling result，不能作为唯一主对比。

新增一键提交入口（inference-only，复用当前 group-RL checkpoints）：

```bash
git pull --ff-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_v2_fair_budget_suite.sh
```

默认提交四个 job：`2p7p@1`、`2p7p@40`、`OOD@1`、`OOD@40`。结果完成后汇总：

```bash
python SketchMol-Understanding-Condition/scripts/collect_direct_smiles_denovo_fair_budget_results.py \
  --suite-root SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_v2_fair_budget_suite_v1
```

如只跑 2p-7p，可设：

```bash
SUCC_DIRECT_FAIR_BUDGET_RUN_OOD=0 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_v2_fair_budget_suite.sh
```

## OOD 结果（n=64，job `16472652`）

| Overall strict | validity | unique valid | 2p bucket strict | 7p bucket strict | mean valid |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.531** | **0.888** | **881 / 1000** | **0.330** | **0.510** | **5.1** |

| vs v1 OOD | overall strict | 7p bucket strict | validity |
| --- | ---: | ---: | ---: |
| v1 n=64 | 0.545 | 0.070 | 0.998 |
| v1 n=256 | 0.801 | 0.190 | 1.000 |
| **v2 n=64** | **0.531** | **0.510** | **0.888** |

OOD overall 略低于 v1 n=64（-1.4pp），但 **7p bucket 大幅提升**（7.0%→51.0%）；validity 降至 88.8%，2p bucket 仅 33.0%。

## 2p7p 结果（n=128 推理-only，job `16484026`）

复用 v2 ckpt（`direct_smiles_denovo_2p7p_v2_mixed_condition/direct_smiles_model/direct_smiles_generator.pt`），不重训。

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **0.965** | **0.914** | **0.857** | **0.775** | **0.648** | **0.584** | **0.791** | **0.997** | **5981 / 6000** |

| vs v2 n=64 | Δ overall strict | 7p strict |
| --- | ---: | ---: |
| n=128 | **+11.0pp**（0.681→0.791） | **+13.5pp**（0.449→0.584） |

ceiling 仍可抬：**79.1%** 已超 v1 best n=256（56.2%）**+22.9pp**；6p/7p 仍低于 SketchMol ref（67.8%/68.5%）。

## OOD 结果（n=128 推理-only，job `16484027`）

复用 OOD v2 ckpt，默认 decoding。

| Overall strict | validity | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: |
| **0.673** | **0.958** | **0.517** | **0.590** |

| vs OOD v2 n=64 | Δ overall strict | 2p bucket | validity |
| --- | ---: | ---: | ---: |
| n=128 | **+14.2pp**（0.531→0.673） | **+18.7pp**（0.330→0.517） | **+7.0pp** |

**sample budget 是 OOD overall 低的主因之一**；n=128 已接近 v1 n=64（54.5%），但仍低于 v1 n=256（80.1%）。

## OOD validity-repair（n=128，保守 decoding，job `16484028`）

不重训；T=**0.70**、top-k=**24**、top-p=**0.90**、repetition penalty=**1.20**（默认 T=0.85 / top-k=40）。

| Overall strict | validity | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: |
| **0.741** | **0.979** | **0.633** | **0.720** |

| vs OOD v2 n=64 | Δ overall strict | validity | 7p bucket |
| --- | ---: | ---: | ---: |
| validity-repair | **+21.0pp**（0.531→0.741） | **+9.1pp**（0.888→0.979） | **+21.0pp**（0.510→0.720） |

**当前 OOD best**：保守 decoding 在不重训下同时修 validity 与 strict；7p bucket **72%** 为所有 OOD 变体最高。

## OOD balanced retrain（job `16484029`）

mixed condition 保持；`property_count_curriculum_loss=0`（只开 sampling 加权，不做 loss 加权）；n=64 推理。

| Overall strict | validity | 2p bucket strict | 7p bucket strict |
| ---: | ---: | ---: | ---: |
| **0.651** | **0.908** | **0.510** | **0.230** |

| vs OOD v2 n=64（full curriculum） | overall strict | 7p bucket | validity |
| --- | ---: | ---: | ---: |
| balanced | **+12.0pp** | **-28.0pp**（0.510→0.230） | **+2.0pp** |

放软 loss curriculum 提升 overall / 2p，但 **7p bucket 崩塌**（51%→23%）；full loss 加权对 rare 7p combo 必要。

## Main pipeline suite（`direct_v2_main_pipeline_0622`，jobs `16509026`–`16509029`）

一键提交 4 变体（`submit_direct_smiles_denovo_v2_rerun_suite.sh`）；复用 v2 base ckpt，n=128 推理-only。总表：`outputs/direct_smiles_denovo_direct_v2_main_pipeline_0622/suite_report.md`。

| Variant | Job | overall strict | validity | 2p | 3p | 4p | 5p | 6p | 7p / 7p bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2p7p default n=128 | `16509026` | **0.791** | 0.997 | 0.965 | 0.914 | 0.857 | 0.775 | 0.648 | 0.584 |
| 2p7p conservative n=128 | `16509027` | **0.776** | 0.993 | 0.952 | 0.898 | 0.848 | 0.735 | 0.652 | 0.571 |
| OOD default n=128 | `16509028` | **0.673** | 0.958 | 0.517 | — | — | — | — | 0.590 |
| OOD conservative n=128 | `16509029` | **0.741** | 0.979 | 0.633 | — | — | — | — | 0.720 |

与首轮 follow-up（`16484026`–`16484028`）**数值一致**（可复现）。

| 对比 | Δ overall strict | 说明 |
| --- | ---: | --- |
| 2p7p conservative vs default | **-1.5pp**（0.791→0.776） | 保守 decoding 对 2p7p **无益**（OOD 则 +6.8pp） |
| OOD conservative vs default | **+6.8pp**（0.673→0.741） | 2p bucket +11.7pp，7p +13.0pp |

**当前 best**：2p7p group RL n=256 **90.9%**；OOD group RL conservative n=256 **89.4%** overall / n=256 default **90.0%** 7p bucket。

## RL v2 初探（job `16532803`）

入口：`submit_direct_smiles_rl_2p7p_v2.sh`。从 v2 SFT ckpt warm-start；1 epoch REINFORCE（rollouts=8，lr=2e-6，sft_weight=1.0）；训练后 n=128 benchmark。

训练：`eval_mean_reward` **0.853**；`mean_reward` 0.399；`sft_loss` 1.23。

2p7p 评测 n=128：

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.718 | 0.534 | 0.391 | 0.266 | 0.195 | 0.139 | **0.374** | 1.000 | **4736 / 6000** |

| vs v2 SFT n=128 | Δ overall strict | 7p strict | unique valid |
| --- | ---: | ---: | ---: |
| RL v1 | **-41.7pp**（0.791→0.374） | **-44.5pp**（0.584→0.139） | 4736（vs 5981） |

优于 v1 RL collapse（23.2%，1642 unique），但 **strict 全面低于 SFT**；reward 上升、strict 崩溃模式仍在。

**运维备注**：评测日志显示 `condition_mixing_mode=features_only`（pipeline bug，已在 `6974f20` 修复）。

## RL v2 condfix rerun（job `16547785`）

`6974f20` 后：`train_direct_smiles_generator_rl.py` 支持 `--condition-mixing-mode` 并从 ckpt 恢复；train + bench 均显式 `append_property_program`。输出目录 `direct_smiles_denovo_2p7p_v2_rl_v2_conditionfix`。

训练：`eval_mean_reward` **1.094**；`sft_loss` **0.947**；评测 `condition_mixing_mode` **append_property_program** ✓。

2p7p 评测 n=128：

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.968 | 0.913 | 0.841 | 0.737 | 0.606 | 0.525 | **0.765** | 1.000 | **5741 / 6000** |

| 对比 | overall strict | Δ vs 上次 RL | 说明 |
| --- | ---: | ---: | --- |
| RL v1（`16532803`，cond bug） | 0.374 | — | 严重低估 |
| **RL condfix（`16547785`）** | **0.765** | **+39.1pp** | conditioning 对齐后恢复 |
| v2 SFT n=128 | 0.791 | — | 仍 best（**-2.6pp** vs RL） |

**结论**：上次 37.4% 的 **绝大部分（~39pp）来自 conditioning mismatch**；对齐后 RL 仅略低于 SFT（-2.6pp），未 collapse，但 **REINFORCE 仍无正向增益**。下一步转 group-relative RL（见 [roadmap](benchmark-roadmap-rl-agentic.md)）。

## Group-relative RL v1（job `16583941`）

入口：`submit_direct_smiles_group_rl_2p7p_v2.sh`（`724e72b`）。从 v2 SFT ckpt warm-start；1 epoch group-relative RL（rollouts=16，`group_zscore`，`reference_kl_weight=0.05`，`sft_weight=1.0`）；`append_property_program`；训练后 n=128 benchmark。输出 `direct_smiles_denovo_2p7p_v2_group_rl_v1/`。

训练：`eval_mean_reward` **-0.176**；`mean_reward` **-0.449**；`sft_loss` **0.808**；`kl_loss` **-0.003**（reward 为负但 strict 仍上升）。

2p7p 评测 n=128：

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.981 | 0.950 | 0.909 | 0.826 | 0.744 | 0.661 | **0.845** | 0.996 | **5970 / 6000** |

| 对比 | overall strict | Δ vs SFT n=128 | 6p / 7p strict | 说明 |
| --- | ---: | ---: | --- | --- |
| v2 SFT n=128 | 0.791 | — | 0.648 / 0.584 | 原 best |
| RL condfix（`16547785`） | 0.765 | -2.6pp | 0.606 / 0.525 | REINFORCE 无增益 |
| **group RL（`16583941`）** | **0.845** | **+5.4pp** | **0.744 / 0.661** | **首次 RL 超 SFT** |
| SketchMol ref | — | — | 0.678 / 0.685 | 6p 已超 ref；7p 仍 -2.4pp |

**结论**：group-relative advantage + reference KL 在 conditioning 对齐前提下 **首次带来正向 strict 增益**（**84.5%**，+5.4pp vs SFT）；高 property bucket（6p/7p）提升最大（+9.6pp / +7.7pp）。

## Group RL n=256 推理缩放（job `16612814`）

复用 `16583941` group RL ckpt；`RUN_TRAIN=0` 推理-only（`8ba87ea`）；`benchmark_num_samples=256`。输出 `benchmark_direct_smiles_group_rl_n256/`。

2p7p 评测 n=256：

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.992 | 0.982 | 0.955 | 0.905 | 0.838 | 0.783 | **0.909** | 0.999 | **5991 / 6000** |

| 对比 | overall strict | Δ vs group RL n=128 | 7p strict | 说明 |
| --- | ---: | ---: | ---: | --- |
| v2 SFT n=128 | 0.791 | — | 0.584 | — |
| group RL n=128（`16583941`） | 0.845 | — | 0.661 | 训练后默认 bench |
| **group RL n=256（`16612814`）** | **0.909** | **+6.4pp** | **0.783** | **当前 best** |
| SketchMol ref | — | — | 0.685 | 7p 已超 ref（**+9.8pp**） |

n=128→n=256 对 group RL ckpt 仍有 **+6.4pp** overall；7p bucket **+12.2pp**（0.661→0.783），**全面超过 SketchMol ref**。SFT n=128→n=256 曾 +11pp（68.1%→79.1%），group RL 缩放增益略小但绝对值更高。

**当前 2p7p best**：group RL n=256 **90.9%**（+11.8pp vs SFT n=128）。可选 follow-up：多 epoch group RL、调 `reference_kl_weight`。

## OOD group-relative RL v1（job `16742519`）

入口：`submit_direct_smiles_group_rl_ood_v2.sh`（`166e969`）。从 OOD v2 SFT ckpt warm-start；1 epoch group-relative RL（rollouts=16，`group_zscore`，`reference_kl_weight=0.05`）；`append_property_program`；训练后 n=128 benchmark。输出 `direct_smiles_denovo_ood_v2_group_rl_v1/`。

训练：`eval_mean_reward` **-0.821**；`mean_reward` **-0.889**；`sft_loss` **0.907**。

OOD 评测 n=128：

| 2p bucket | 7p bucket | overall strict | validity | unique valid |
| ---: | ---: | ---: | ---: | ---: |
| 0.620 | 0.590 | **0.756** | 0.952 | **944 / 1000** |

| 对比 | overall strict | 2p bucket | 7p bucket | validity |
| --- | ---: | ---: | ---: | ---: |
| OOD SFT n=128 default | 0.673 | 0.517 | 0.590 | 0.958 |
| OOD conservative n=128 | 0.741 | 0.633 | **0.720** | **0.979** |
| **OOD group RL（`16742519`）** | **0.756** | 0.620 | 0.590 | 0.952 |

**结论**：group RL 将 OOD overall strict 提升至 **75.6%**（**+8.3pp** vs SFT default，**+1.5pp** vs conservative），但 7p bucket **59.0%** 仍低于 conservative **72.0%**（-13.0pp）。conservative decoding 与 n=256 缩放可分别修复 7p 与 overall ceiling（见下）。

## OOD group RL follow-up（P0，`16768168`–`16768170`）

复用 `16742519` group RL ckpt；`RUN_TRAIN=0`（`8ba87ea` + `RUN_SCRIPT` 修复）。入口见 `paper-benchmark-plan.md`。

| Job | 变体 | overall strict | 2p bucket | 7p bucket | validity |
| --- | --- | ---: | ---: | ---: | ---: |
| `16742519` | default n=128 | 0.756 | 0.620 | 0.590 | 0.952 |
| **`16768168`** | **conservative n=128** | **0.802** | 0.663 | **0.780** | **0.970** |
| **`16768169`** | **default n=256** | **0.890** | 0.810 | **0.900** | 0.990 |
| **`16768170`** | **conservative n=256** | **0.894** | **0.827** | 0.870 | 0.989 |

| 对比 | Δ overall vs `16742519` | Δ 7p vs `16742519` | 说明 |
| --- | ---: | ---: | --- |
| conservative n=128 | **+4.6pp** | **+19.0pp** | 同时超 conservative SFT（0.741 / 0.720） |
| default n=256 | **+13.4pp** | **+31.0pp** | sample budget 是 OOD 主瓶颈之一 |
| conservative n=256 | **+13.8pp** | **+28.0pp** | **OOD overall best** |

**结论**：
1. **7p 短板来自 default decoding**，非 group RL 训练本身；conservative n=128 即把 7p 从 59.0% 拉到 **78.0%**（+19.0pp）。
2. **n=256 大幅抬 ceiling**：default n=256 overall **89.0%**，7p **90.0%**。
3. **主文 OOD 主结果建议**：conservative n=256 **89.4%** overall（best overall）；若强调 7p bucket 用 default n=256 **90.0%**。

## num_samples 缩放（v2 ckpt，推理-only）

| variant | 2p7p strict | OOD strict | OOD validity | OOD 7p bucket |
| --- | ---: | ---: | ---: | ---: |
| v2 n=64（重训评测） | 0.681 | 0.531 | 0.888 | 0.510 |
| v2 2p7p n=128 default | **0.791** | — | — | — |
| v2 2p7p n=128 conservative | 0.776 | — | — | — |
| v2 OOD n=128 | — | 0.673 | 0.958 | 0.590 |
| v2 OOD validity-repair n=128 | — | 0.741 | 0.979 | 0.720 |
| v2 OOD group RL n=128（`16742519`） | — | 0.756 | 0.952 | 0.590 |
| v2 OOD group RL conservative n=128（`16768168`） | — | 0.802 | 0.970 | 0.780 |
| v2 OOD group RL n=256（`16768169`） | — | 0.890 | 0.990 | 0.900 |
| v2 OOD group RL conservative n=256（`16768170`） | — | **0.894** | 0.989 | 0.870 |
| v2 OOD balanced n=64（重训） | — | 0.651 | 0.908 | 0.230 |
| v2 RL n=128（cond bug，`16532803`） | 0.374 | — | — | — |
| v2 RL n=128 condfix（`16547785`） | 0.765 | — | — | — |
| v2 group RL n=128（`16583941`） | **0.845** | — | — | — |
| v2 group RL n=256（`16612814`） | **0.909** | — | — | — |

## 与 v1 / hybrid 对照

| 方法 | 2p7p strict | OOD strict | retrieval? |
| --- | ---: | ---: | --- |
| direct v1 n=256 | 0.562 | 0.801 | 否 |
| direct v1 DPO n=256 | 0.521 | — | 否 |
| direct v2 n=64 | 0.681 | 0.531 | 否 |
| **direct v2 2p7p n=128 SFT** | 0.791 | — | 否 |
| **direct v2 group RL n=128** | 0.845 | — | 否 |
| **direct v2 group RL n=256** | **0.909** | — | 否 |
| **direct v2 OOD group RL conservative n=256** | — | **0.894** | 否 |
| direct v2 OOD group RL n=256 | — | 0.890 | 否 |
| direct v2 OOD group RL conservative n=128 | — | 0.802 | 否 |
| direct v2 OOD group RL n=128 | — | 0.756 | 否 |
| direct v2 OOD validity-repair n=128 | — | 0.741 | 否 |
| direct v2 OOD balanced n=64 | — | 0.651 | 否 |
| direct v2 RL n=128（cond bug） | 0.374 | — | 否 |
| direct v2 RL n=128 condfix | 0.765 | — | 否 |
| direct v1 RL n=256 | 0.232 | — | 否 |
| hybrid latent_property_rerank | 0.970 | 0.985 | 是 |
| SketchMol ref（2p） | 0.804 | — | — |

## 结论

1. **mixed condition + curriculum 有效**：2p7p n=64 **68.1%**；n=128 推理-only 达 **79.1%**（+11pp），显著优于 v1 全部规模。
2. **OOD sample budget 不够是主因之一**：n=64→128 overall **+14.2pp**（53.1%→67.3%）；validity-repair decoding 进一步到 **74.1%**。
3. **OOD group RL + conservative / n=256 刷新 best**：conservative n=256 overall **89.4%**；default n=256 7p **90.0%**；conservative n=128 即修复 7p（59%→78%）。
4. **balanced curriculum 不可行**：关 loss 加权后 7p bucket **23%**（vs full 51%）；rare combo 依赖 loss curriculum。
5. **Node Fail 可恢复**：`16472651` epoch 9 中断后 resume 补完。
6. **main pipeline suite 可复现**：`direct_v2_main_pipeline_0622` 四变体与首轮 follow-up 一致；2p7p 用 default decoding，OOD 用 conservative。
7. **RL condfix 验证 conditioning bug**：对齐后 strict **76.5%**（vs 错误 pipeline 37.4%，**+39.1pp**）；仍略低于 SFT 79.1%（-2.6pp），REINFORCE 无增益。
8. **group-relative RL 首次超 SFT**：`16583941` strict **84.5%**（**+5.4pp** vs SFT）；6p/7p +9.6pp/+7.7pp。
9. **group RL n=256 达 90.9%**：`16612814` 推理-only（+6.4pp vs n=128）；7p **78.3%** 超 SketchMol ref；**当前 2p7p best**。
10. **OOD group RL baseline 75.6%**：`16742519`；7p 弱于 conservative 主因是 default decoding。
11. **P0 follow-up 完成**：conservative n=128 **80.2%/78.0%** 7p；n=256 **89.0%/90.0%** 7p；conservative n=256 **89.4%** overall best。
12. **下一步**：MuMO / C-MuMO port（P1）；多 epoch group RL；Table1 vs attack 并排量化 selection gain。

## 提交命令（归档）

```bash
# 2p7p v2（默认 mixed condition + curriculum，n=64 推理）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_benchmark.sh

# OOD v2
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_benchmark.sh

# 2p7p resume（Node Fail 后）
SUCC_DIRECT_DENOVO_RESUME_CHECKPOINT=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/direct_smiles_model/latest_checkpoint.pt \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_benchmark.sh

# 2p7p v2 n=128 推理-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_2p7p_v2_n128_benchmark.sh

# OOD v2 n=128 推理-only
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_n128_benchmark.sh

# OOD v2 validity-repair（保守 decoding，不重训）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_validity_repair_benchmark.sh

# OOD v2 balanced retrain（curriculum 只开 sampling）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_ood_v2_balanced_benchmark.sh

# main pipeline suite（4 变体一键提交）
SUCC_DIRECT_V2_SUITE_TAG=direct_v2_main_pipeline_0622 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_denovo_v2_rerun_suite.sh

# 收 suite 总表
python3 SketchMol-Understanding-Condition/scripts/collect_direct_smiles_denovo_v2_suite_results.py \
  --suite-root SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_direct_v2_main_pipeline_0622

# v2 conservative RL 2p7p（1 epoch + bench n=128）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_rl_2p7p_v2.sh

# v2 group-relative RL 2p7p（1 epoch rollouts=16 + bench n=128）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_2p7p_v2.sh

# OOD group RL follow-up（P0，benchmark-only）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_benchmark.sh
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_n256_benchmark.sh
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2_conservative_n256_benchmark.sh

# v2 group-relative RL OOD（1 epoch rollouts=16 + bench n=128）
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2.sh

# OOD group RL benchmark-only 复跑
SUCC_DIRECT_OOD_GROUP_RL_RUN_TRAIN=0 \
SUCC_DIRECT_OOD_GROUP_RL_RL_CHECKPOINT=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_ood_v2.sh

# group RL n=256 benchmark（复用 ckpt，不重训）
SUCC_DIRECT_GROUP_RL_RUN_TRAIN=0 \
SUCC_DIRECT_GROUP_RL_BENCHMARK_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/benchmark_direct_smiles_group_rl_n256 \
SUCC_DIRECT_GROUP_RL_BENCHMARK_MODEL_DIR=SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl_eval_n256 \
SUCC_DIRECT_GROUP_RL_BENCHMARK_NUM_SAMPLES=256 \
SUCC_DIRECT_GROUP_RL_SLURM_TIME=16:00:00 \
SUCC_DIRECT_GROUP_RL_SLURM_JOB_NAME=succ-direct-smiles-2p7p-v2-group-rl-n256 \
bash SketchMol-Understanding-Condition/scripts/submit_direct_smiles_group_rl_2p7p_v2.sh
```

产出：
- `outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/`
- `outputs/direct_smiles_denovo_ood_v2_mixed_condition/`
- `outputs/direct_smiles_denovo_2p7p_v2_mixed_condition_n128/`
- `outputs/direct_smiles_denovo_ood_v2_mixed_condition_n128/`
- `outputs/direct_smiles_denovo_ood_v2_mixed_condition_validity_repair_n128/`
- `outputs/direct_smiles_denovo_ood_v2_mixed_condition_balanced/`
- `outputs/direct_smiles_denovo_direct_v2_main_pipeline_0622/`（suite：`suite_report.md` / `suite_summary.csv`）
- `outputs/direct_smiles_denovo_2p7p_v2_rl_v1/`
- `outputs/direct_smiles_denovo_2p7p_v2_rl_v2_conditionfix/`
- `outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/`
- `outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/benchmark_direct_smiles_group_rl_n256/`
- `outputs/direct_smiles_denovo_ood_v2_group_rl_v1/`
- `outputs/direct_smiles_denovo_ood_v2_group_rl_v1/benchmark_direct_smiles_group_rl_conservative_n128/`
- `outputs/direct_smiles_denovo_ood_v2_group_rl_v1/benchmark_direct_smiles_group_rl_n256/`
- `outputs/direct_smiles_denovo_ood_v2_group_rl_v1/benchmark_direct_smiles_group_rl_conservative_n256/`
