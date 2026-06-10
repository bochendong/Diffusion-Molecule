# SUCC UniVideo × MolEdit-Instruct v2 Fix

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + materialized benchmark + table metrics） |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Understanding-Condition` |
| **训练 Job** | `15866598`（`succ-univideo-mol`） |
| **Benchmark Job** | `15866600`（`succ-univideo-bench`） |
| **Table Job** | `15866601`（`succ-moledit-table`） |
| **基线** | v1: [moledit-instruct-v1.md](moledit-instruct-v1.md) |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15866598` | `succ-univideo-mol` | **27m16s** | 0 |
| `15866600` | `succ-univideo-bench` | **2m37s** | 0 |
| `15866601` | `succ-moledit-table` | **5s** | 0 |

## 目的

修复 v1 问题：Table1-balanced 训练/eval、weighted active-property loss、MW/SA/RB 加权监督。

## 配置要点

| 变量 | 值 |
| --- | --- |
| 输出 | `outputs/univideo_molecule_generation_moledit_instruct_v2_fix/` |
| Table1-only balanced | train 5000/task，eval 100/task |
| 采样 | `weighted`，`MW=4,SA=3,RB=2` |
| aux loss | `MW=3,SA=3,RB=2`，weight=0.50 |
| residual_sample_scale | 1.25，connector_latent_blend=0.10 |

## 训练结果（eval_latent，n=395 Table1-balanced）

| 指标 | v1 | v2 fix |
| --- | ---: | ---: |
| latent MAE | 0.735 | 0.738 |
| source latent cosine | 0.991 | **0.993** |
| target latent cosine | 0.380 | 0.374 |

Latent 仍高度贴 source；property 改善主要体现在 table/benchmark 层。

## Materialized Benchmark（job `15866600`，n=395）

### 2p–7p strict（`edit_latent_source_similarity_rerank`）

| 2p | 3p | 4p | 5p | 6p | 7p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.727** | — | 0.800 | 0.826 | 0.765 | 0.519 |

v1 对比：2p **0.450** → **0.727**

### Source-similarity

| mean Tani | strict@0.4 | strict@0.6 |
| ---: | ---: | ---: |
| **0.580** | **0.729** | 0.729 |

v1：mean Tani 0.428，strict@0.4 0.379

## MolEdit Table Metrics（job `15866601`）

| task | Acc_all(0.65) | Acc_all(0.15) | n | v1 Acc(0.65) | MolEditRL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rotbonds↓ | **0.740** | 0.920 | 100 | 0.333 | 0.634 |
| MW↑ | **0.631** | 0.750 | 84 | 0.000 | 0.404 |
| SA↓ | **0.480** | 0.590 | 100 | 0.333 | 0.628 |
| Haccept↓ LogP↑ | 1.000 | 1.000 | 1 | — | 0.316 |
| **overlap mean** | **0.617** | **0.753** | — | 0.222 | 0.555 |

MW↑ 从 0 提升到 **0.631**；overlap mean Acc(0.65) 从 0.222 提升到 **0.617**，接近 MolEditRL 0.555 水平。GSK3B/DRD2 仍因无 PyTDC 跳过。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix/
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/eval_latent/
  univideo_molecule/benchmark_materialized_primary_fast/
  univideo_molecule/moledit_table_metrics/
```

日志：
- `logs/succ-univideo-mol-15866598.log`
- `logs/succ-univideo-bench-15866600.log`
- `logs/succ-moledit-table-15866601.log`

## 结论与下一步

1. **Table1-balanced + weighted loss 有效**：MW↑/Rotbonds↓/SA↓ 全面提升，overlap mean 接近 MolEditRL。
2. **Materialized 2p strict 0.73**，source Tani 0.58，比 v1 更平衡。
3. **Latent cosine 仍贴 source**（0.993），edit signal 在 latent 层未明显改善。
4. **对照 Unified source-anchored v2**：见 [moledit-sourceanchored-v2.md](../unified-3m/moledit-sourceanchored-v2.md)；完整 Table1 需 PyTDC。
