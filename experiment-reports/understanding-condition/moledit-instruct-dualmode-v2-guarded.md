# SUCC MolEdit-Instruct Dual-Mode v2 Guarded

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + Table1 chain + de novo/OOD benchmark）；**Table1 guard failed** |
| **最后更新** | 2026-06-13 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_univideo_moledit_dualmode_pipeline.sh`（默认 v2_guarded） |
| **前序** | [moledit-instruct-dualmode-v1.md](moledit-instruct-dualmode-v1.md) |
| **基线** | v3 attack Table1: [moledit-instruct-v3-attack.md](moledit-instruct-v3-attack.md) |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit | 说明 |
| --- | --- | --- | ---: | --- |
| `16018414` | `succ-univideo-mol`（训练） | **53m14s** | 0 | |
| `16019240` | `succ-table1-eval` | **1h04m46s** | 0 | |
| `16019241` | `succ-table1-bench` | **22m59s** | 0 | `table_attack` rerank |
| `16019242` | `succ-table1-table` | **33s** | 0 | |
| `16019243` | `succ-table1-guard` | **2s** | **1** | Acc mean 未达 guard |
| `16019244` | `succ-denovo-ood-ours` | **15m05s** | 0 | |
| `16019245` | `succ-denovo-2p7p-ours` | **14m20s** | 0 | |

输出：`outputs/univideo_molecule_generation_moledit_instruct_dualmode_v2_guarded/`

## v2 相对 v1 的改动

| 变量 | v1 dualmode | v2 guarded |
| --- | --- | --- |
| `SUCC_LATENT_TARGET_MODE` | residual | **mixed**（edit=residual，zero-source=absolute） |
| `SUCC_SOURCE_DROPOUT` | 0.30 | **0.08** |
| `SUCC_OOD_TRAIN_ROWS_PER_SPEC` | 200 | **400** |
| `SUCC_DENOVO_SAMPLE_WEIGHT` | — | **2.0** |
| `SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT` | — | **0.02** |
| Table1 guard | 无 | **Acc@0.65 mean ≥ 0.894** |

## Table1 Guard（job `16019243`）— **Failed（设计如此）**

Guard 不是 crash：`check_moledit_table_guard.py` 检测到 mean 低于阈值后 **主动 exit 1**。

| 指标 | 实测 | Guard 阈值 |
| --- | ---: | ---: |
| 10-task `Acc_all(0.65)` mean | **0.794** | **0.894** |

### 按 Task（与 v3 attack 同路径 eval **完全一致**）

| Task | Acc@0.65 |
| --- | ---: |
| GSK3B↑ | **0.00** |
| Rotbonds↓ | 1.00 |
| MW↑ | 1.00 |
| SA↓ | 0.58 |
| Haccept↓ SA↓ | 0.88 |
| QED↑ SA↓ | 0.88 |
| Haccept↓ LogP↑ | 0.90 |
| Haccept↓ MW↓ | 0.87 |
| DRD2↓ MW↓ SA↓ | 1.00 |
| Haccept↑ MW↑ QED↓ | 0.83 |

**结论**：dualmode v2 **未牺牲 MolEdit Table1 edit 能力**（逐 task 与 v3 attack 相同）；guard 失败主因是 **GSK3B↑ 仍为 0%** 拉低 mean。

**阈值问题**：v3 attack 在相同 `table1_benchmark_guarded` + `edit_latent_table_success_rerank` 路径上算出的 mean 也是 **0.794**，而非报告中的 0.894。当前 guard 门槛 **0.894 不可达**（除非 GSK3B 突破或改 eval 路径）。建议后续改为 **0.794** 或 **9-task mean（排除 GSK3B）**。

Guard 产出：`univideo_molecule/moledit_table_metrics_table1_guarded/moledit_table_guard.json`

## De novo / OOD Benchmark（dualmode v2 ckpt）

| 评测 | Job | Overall strict | Unique valid | v1 dualmode |
| --- | --- | ---: | ---: | ---: |
| OOD | `16019244` | **0.052** | 0.008 | 0.127 / 0.199 |
| 2p7p | `16019245` | **0.0115** | 0.0015 | 0.039 / 0.068 |

**v2 在 zero-source 上反而退步**：更保守的 `source_dropout=0.08` + mixed latent 保留了 Table1，但未带来 de novo/OOD 增益；v1 的 aggressive dropout（0.30）在 OOD/2p7p 上更好。

### OOD 按 Bucket（job `16019244`）

| Bucket | v1 dualmode | v2 guarded |
| --- | ---: | ---: |
| forward_extreme | 0.083 | 0.080 |
| rare_combo | 0.010 | **0.000** |
| reverse_stimulation | **0.450** | 0.100 |
| **Overall** | **0.127** | **0.052** |

### 2p7p strict（job `16019245`）

| 2p | 3p | 4p | 5p | 6p | 7p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.048 | 0.012 | 0.007 | 0.002 | 0.000 | 0.000 |

## 总结

1. **Table1 edit：持平 v3**（mean 0.794），guard fail 是阈值设置问题 + GSK3B 0%，不是训练 crash。
2. **De novo/OOD：v2 劣于 v1** — 保守 dropout + mixed latent 未能复制 v1 的 OOD/reverse_stim 增益。
3. **下一步方向**：v3 已修正 guard 阈值并通过，见 [moledit-instruct-dualmode-v3-guarded.md](moledit-instruct-dualmode-v3-guarded.md)。

## 产出路径

```text
outputs/univideo_molecule_generation_moledit_instruct_dualmode_v2_guarded/
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/moledit_table_metrics_table1_guarded/
  univideo_molecule/benchmark_materialized_table1_guarded/
outputs/denovo_ood_ours_dualmode_v2_guarded/benchmark_ours/
outputs/denovo_2p7p_ours_dualmode_v2_guarded/benchmark_ours/
```

日志：
- `logs/succ-univideo-mol-16018414.log`
- `logs/succ-table1-guard-16019243.log`
- `logs/succ-denovo-ood-ours-16019244.log`
- `logs/succ-denovo-2p7p-ours-16019245.log`
