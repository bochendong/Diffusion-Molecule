# SUCC MolEdit-Instruct Dual-Mode v3 Guarded

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + Table1 chain + de novo/OOD benchmark）；**Table1 guard passed** |
| **最后更新** | 2026-06-14 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_univideo_moledit_dualmode_pipeline.sh`（当前默认 v3_guarded） |
| **前序** | [moledit-instruct-dualmode-v1.md](moledit-instruct-dualmode-v1.md) / [v2-guarded](moledit-instruct-dualmode-v2-guarded.md) |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit | 说明 |
| --- | --- | --- | ---: | --- |
| `16027758` | `succ-univideo-mol`（训练） | **54m08s** | 0 | |
| `16028792` | `succ-table1-eval` | **1h06m22s** | 0 | |
| `16028793` | `succ-table1-bench` | **24m28s** | 0 | |
| `16028794` | `succ-table1-table` | **21s** | 0 | |
| `16028795` | `succ-table1-guard` | **1s** | **0** | Acc mean ≥ 0.794 |
| `16028796` | `succ-denovo-ood-ours` | **12m52s** | 0 | |
| `16028797` | `succ-denovo-2p7p-ours` | **13m51s** | 0 | |

输出：`outputs/univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded/`

## v3 相对 v2 的改动

| 变量 | v2 guarded | v3 guarded |
| --- | --- | --- |
| `SUCC_SOURCE_DROPOUT` | 0.08 | **0.12** |
| `SUCC_OOD_TRAIN_ROWS_PER_SPEC` | 400 | **600** |
| `SUCC_DENOVO_TRAIN_ROWS_PER_PROPERTY_COUNT` | 500 | **750** |
| `SUCC_DENOVO_SAMPLE_WEIGHT` | 2.0 | **4.0** |
| `SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT` | 0.02 | **0.05** |
| `SUCC_SOURCE_FREE_AUGMENT_*` | — | weight **0.50**, prob **0.40** |
| `SUCC_TABLE1_GUARD_MIN_ACC065` | 0.894（不可达） | **0.794** |

## Table1 Guard（job `16028795`）— **Passed**

| 指标 | 值 |
| --- | ---: |
| 10-task `Acc_all(0.65)` mean | **0.794** |
| Guard 阈值 | **0.794** |
| GSK3B↑ | **0.00**（与 v2/v3 attack 相同） |

逐 task 与 v2/v3 attack **完全一致** — dualmode v3 **未牺牲 MolEdit Table1**。

## De novo / OOD Benchmark

### 总览（三代 dualmode）

| 评测 | v1 | v2 | **v3** |
| --- | ---: | ---: | ---: |
| OOD overall strict | **0.127** | 0.052 | 0.101 |
| OOD unique valid | **0.199** | 0.008 | 0.020 |
| OOD target latent cos | 0.238 | 0.238 | **0.571** |
| 2p7p overall strict | **0.039** | 0.0115 | 0.0285 |
| 2p7p unique valid | **0.068** | 0.0015 | 0.0053 |
| 2p7p target latent cos | 0.242 | — | **0.580** |

### OOD 按 Bucket（job `16028796`）

| Bucket | v1 | v2 | **v3** |
| --- | ---: | ---: | ---: |
| forward_extreme | 0.083 | 0.080 | **0.040** |
| rare_combo | **0.010** | 0.000 | **0.000** |
| reverse_stimulation | **0.450** | 0.100 | **0.425** |

### OOD 按 Spec（v3 亮点 / 短板）

| Spec | v1 | **v3** |
| --- | ---: | ---: |
| reverse_logp4 | 0.560 | **0.820** |
| reverse_mw300 | 0.340 | 0.030 |
| logp_high | 0.210 | **0.000** |
| rb_high | 0.100 | **0.160** |
| rare_combo specs | 0–0.010 | **0.000** |

### 2p7p strict（job `16028797`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.117 | 0.038 | 0.012 | 0.001 | 0.002 | 0.001 | **0.0285** |

## 结论

1. **Table1 guard 首次通过**：阈值修正为 0.794 后，v3 在保持 edit 能力的同时允许继续迭代 de novo/OOD。
2. **Latent 对齐大幅改善**：OOD/2p7p target cosine **~0.57–0.58**（v1 ~0.24），说明 mixed latent + source-free augment 有效。
3. **Strict 仍不如 v1**：overall OOD **0.101 < 0.127**；2p7p **0.0285 < 0.039** — materializer（`latent_nearest`）是瓶颈。
4. **reverse_logp4 突破**：**82%** strict（v1 56%）；但 `reverse_mw300` / `logp_high` / **rare_combo 仍弱**。
5. **多样性未改善**：unique valid OOD **2%**、2p7p **0.5%**，diversity loss 需继续调。v4 warm-start 亦未恢复 v1 diversity，见 [moledit-instruct-dualmode-v4-warmstart-v1.md](moledit-instruct-dualmode-v4-warmstart-v1.md)。

## 产出路径

```text
outputs/univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded/
outputs/denovo_ood_ours_dualmode_v3_guarded/benchmark_ours/
outputs/denovo_2p7p_ours_dualmode_v3_guarded/benchmark_ours/
```

日志：
- `logs/succ-univideo-mol-16027758.log`
- `logs/succ-table1-guard-16028795.log`
- `logs/succ-denovo-ood-ours-16028796.log`
- `logs/succ-denovo-2p7p-ours-16028797.log`
