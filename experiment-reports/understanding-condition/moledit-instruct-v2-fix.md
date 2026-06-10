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

## MolEdit Table Metrics（job `15897034`，PyTDC 修复后）

| task | Acc_all(0.65) | Acc_all(0.15) | n | MolEditRL Acc(0.65) |
| --- | ---: | ---: | ---: | ---: |
| **GSK3B↑** | 0.000 | 0.000 | 100 | 0.342 |
| Rotbonds↓ | **0.740** | 0.920 | 100 | 0.634 |
| MW↑ | **0.631** | 0.750 | 84 | 0.404 |
| SA↓ | **0.480** | 0.590 | 100 | 0.628 |
| Haccept↓ LogP↑ | 1.000 | 1.000 | 1 | 0.316 |
| **DRD2↓ MW↓ SA↓** | **0.500** | 0.800 | 10 | 0.518 |
| **6-task mean** | **0.559** | **0.677** | — | — |
| overlap-3 mean (RB/MW/SA) | **0.617** | **0.753** | — | 0.555 |

PyTDC 修复：`rdkit.six` shim + `module load rdkit` + `setup_moledit_tdc.sh`。此前 job `15866601` 仅 4 行（缺 GSK3B/DRD2）。

benchmark_decoded 仅覆盖 **6/10** Table1 task；其余 4 个组合 task 在预测集中无行。

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
- `logs/succ-moledit-table-15866601.log`（旧，缺 TDC）
- `logs/succ-moledit-table-15897034.log`（PyTDC 完整版）

## 结论与下一步

1. **Table1-balanced + weighted loss 有效**：6-task mean Acc(0.65)=**0.559**，overlap-3 mean **0.617**（仍高于 MolEditRL 0.555）。
2. **GSK3B↑ 仍弱**（Acc=0），DRD2 组合 task Acc(0.65)=0.5；后续可加强 GSK3B 方向监督。
3. **Materialized 2p strict 0.73**，source Tani 0.58，比 v1 更平衡。
4. **对照 Unified source-anchored v2**：见 [moledit-sourceanchored-v2.md](../unified-3m/moledit-sourceanchored-v2.md)。
