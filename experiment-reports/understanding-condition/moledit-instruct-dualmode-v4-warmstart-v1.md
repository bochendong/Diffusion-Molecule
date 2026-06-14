# SUCC MolEdit-Instruct Dual-Mode v4 Warmstart-v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + Table1 chain + de novo/OOD benchmark）；**Table1 guard passed**；zero-source strict **未达目标** |
| **最后更新** | 2026-06-14 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_univideo_moledit_dualmode_pipeline.sh`（当前默认 v4_warmstart_v1） |
| **前序** | [dualmode-v1](moledit-instruct-dualmode-v1.md) / [dualmode-v3-guarded](moledit-instruct-dualmode-v3-guarded.md) |

## 动机

v3 guarded 保住了 Table1 guard，并把 OOD / 2p7p target latent cosine 拉到
0.57–0.58；但 strict success 和 unique valid 仍低于 v1。v4 从 dualmode v1
checkpoint warm-start，再用更小学习率（3e-4）和短 schedule（1+8+4）吸收 v3
的 mixed-latent 与 source-free augmentation 经验。

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit | 说明 |
| --- | --- | --- | ---: | --- |
| `16042564` | `succ-univideo-mol`（训练） | **52m00s** | 0 | warm-start from v1 |
| `16043202` | `succ-table1-eval` | **1h05m29s** | 0 | |
| `16043203` | `succ-table1-bench` | **24m45s** | 0 | |
| `16043204` | `succ-table1-table` | **21s** | 0 | |
| `16043205` | `succ-table1-guard` | **1s** | **0** | Acc mean ≥ 0.794 |
| `16043206` | `succ-denovo-ood-ours` | **12m06s** | 0 | |
| `16043207` | `succ-denovo-2p7p-ours` | **13m48s** | 0 | |

输出：`outputs/univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1/`

## 默认配置

| 变量 | v3 guarded | v4 warmstart-v1 |
| --- | ---: | ---: |
| `SUCC_INIT_CHECKPOINT` | none | `...dualmode_v1/.../univideo_molecule_generation.pt` |
| `SUCC_INIT_STATS_FROM_CHECKPOINT` | 0 | **1** |
| `SUCC_LR` | 1e-3 | **3e-4** |
| `SUCC_STAGE1/2/3_EPOCHS` | 4 / 14 / 6 | **1 / 8 / 4** |
| `SUCC_SOURCE_DROPOUT` | 0.12 | 0.12 |
| `SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT` | 0.05 | **0.08** |
| `SUCC_TABLE1_GUARD_MIN_ACC065` | 0.794 | 0.794 |

## Table1 Guard（job `16043205`）— **Passed**

| 指标 | 值 |
| --- | ---: |
| Acc@0.65 mean | **0.794** |
| GSK3B↑ | **0.00** |

逐 task 与 v2/v3/v3 attack **完全一致** — warm-start 短训未牺牲 MolEdit edit。

## 结果 vs 目标

| 指标 | 目标 | v1 | v3 | **v4** | 达成？ |
| --- | --- | ---: | ---: | ---: | --- |
| Table1 guard | pass | — | ✅ | **✅** | ✅ |
| OOD overall strict | > v1 0.127 | **0.127** | 0.101 | **0.097** | ❌ |
| OOD unique valid | ≈ v1 0.199 | **0.199** | 0.020 | **0.012** | ❌ |
| OOD target latent cos | ≈ v3 0.571 | 0.238 | 0.571 | **0.574** | ✅ |
| 2p7p overall strict | > v1 0.039 | **0.039** | 0.0285 | **0.0118** | ❌ |
| 2p7p unique valid | ≈ v1 0.068 | **0.068** | 0.0053 | **0.0037** | ❌ |
| 2p7p target latent cos | ≈ v3 | 0.242 | 0.580 | **0.584** | ✅ |

## OOD 按 Bucket（job `16043206`）

| Bucket | v1 | v3 | **v4** |
| --- | ---: | ---: | ---: |
| forward_extreme | 0.083 | 0.040 | **0.102** |
| rare_combo | 0.010 | 0.000 | **0.000** |
| reverse_stimulation | **0.450** | 0.425 | **0.280** |

### Spec 级 trade-off（v4）

| Spec | v1 | v3 | **v4** |
| --- | ---: | ---: | ---: |
| reverse_logp4 | 0.560 | **0.820** | **0.060** |
| reverse_mw300 | 0.340 | 0.030 | **0.500** |
| rb_high | 0.100 | 0.160 | **0.410** |
| logp_high | 0.210 | 0.000 | 0.000 |

v4 丢失 v3 的 `reverse_logp4` 优势，换来 `reverse_mw300` / `rb_high` 提升，但 reverse bucket 整体下降。

## 2p7p strict（job `16043207`）

| 2p | 3p | 4p | 5p | 6p | 7p | overall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.052 | 0.015 | 0.003 | 0.001 | 0.000 | 0.000 | **0.0118** |

Latent cos **0.584**（三代最高），strict **三代最低** — materializer 瓶颈更明显。

## 结论

1. **Warm-start 假设未成立**：v1 diversity / reverse stim 未继承；latent 对齐接近 v3，strict 低于 v1 和 v3。
2. **Table1 仍安全**：短 fine-tune + v1 init 足够保住 guard。
3. **当前 best zero-source checkpoint 仍是 v1**（OOD 0.127，2p7p 0.039）。
4. **下一步**：v1 warm-start + 更高 source_dropout（0.25–0.30）；或停止改训练、改 materializer / guidance；rare_combo 需专项 oversample。

## 产出路径

```text
outputs/univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1/
outputs/denovo_ood_ours_dualmode_v4_warmstart_v1/benchmark_ours/
outputs/denovo_2p7p_ours_dualmode_v4_warmstart_v1/benchmark_ours/
```

日志：
- `logs/succ-univideo-mol-16042564.log`
- `logs/succ-table1-guard-16043205.log`
- `logs/succ-denovo-ood-ours-16043206.log`
- `logs/succ-denovo-2p7p-ours-16043207.log`
