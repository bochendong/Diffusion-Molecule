# SUCC UniVideo × MolEdit-Instruct v2 Fix

| 字段 | 值 |
| --- | --- |
| **状态** | completed（含 Table1 扩展 7/10 task） |
| **最后更新** | 2026-06-10 |
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

## Table1 扩展 benchmark（2026-06-10，completed）

从 train+eval 按 task 采样（eval 优先，每 task 最多 100），补全 eval 子集未覆盖的 train 行。

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15898378` | `succ-table1-eval` | **19m37s** | 0 |
| `15898537` | `succ-table1-bench` | **4m06s** | 0 |
| `15898540` | `succ-table1-table` | **30s** | 0 |

Pack：`dataset/table1_benchmark/`（**510 行，7/10 task**）。3 个组合 task 在 enhanced 数据集中为 0 行，无法达到 MolEditRL 完整 10-task。

### Table1 eval_latent（n=510）

| 指标 | eval 子集 (n=395) | Table1 扩展 (n=510) |
| --- | ---: | ---: |
| latent MAE | 0.738 | 0.728 |
| source latent cosine | 0.993 | 0.994 |
| target latent cosine | 0.374 | 0.382 |

### Materialized（job `15898537`，n=510，`edit_latent_source_similarity_rerank`）

| 2p strict | mean source Tani | strict@0.4 |
| ---: | ---: | ---: |
| **0.778** | 0.519 | 0.592 |

对比 eval 子集（n=395）：2p **0.727→0.778**，mean Tani **0.580→0.519**（补入 train 行后 source 保持变难）。

### MolEdit Table Metrics（job `15898540`，7 task）

| task | Acc_all(0.65) | Acc_all(0.15) | n | eval子集 Acc(0.65) |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.000 | 0.000 | 100 | 0.000 |
| Rotbonds↓ | 0.560 | 0.850 | 100 | **0.740** |
| MW↑ | 0.400 | 0.630 | 100 | **0.631** (n=84) |
| SA↓ | 0.410 | 0.590 | 100 | 0.480 |
| QED↑ SA↓ | 0.000 | 1.000 | 1 | — |
| Haccept↓ LogP↑ | 0.778 | 0.778 | 9 | 1.000 (n=1) |
| DRD2↓ MW↓ SA↓ | **0.700** | 0.930 | 100 | 0.500 (n=10) |
| **7-task mean** | **0.407** | **0.540** | — | — |
| overlap-3 (RB/MW/SA) | **0.457** | **0.690** | — | 0.617 |

补全 train 行后：**DRD2 组合 task 明显提升**（0.50→0.70），但 RB/MW/SA 在满 100 行上回落；此前 eval 子集数字（overlap-3 **0.617**）部分受益于更小、更偏 eval 的样本池。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix/
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/eval_latent/
  univideo_molecule/benchmark_materialized_primary_fast/
  univideo_molecule/moledit_table_metrics/
  dataset/table1_benchmark/
  univideo_molecule/table1_eval_latent/
  univideo_molecule/benchmark_materialized_table1_primary_fast/
  univideo_molecule/moledit_table_metrics_table1/
```

日志：
- `logs/succ-univideo-mol-15866598.log`
- `logs/succ-univideo-bench-15866600.log`
- `logs/succ-moledit-table-15866601.log`（旧，缺 TDC）
- `logs/succ-moledit-table-15897034.log`（PyTDC 完整版，eval 子集）
- `logs/succ-table1-eval-15898378.log`
- `logs/succ-table1-bench-15898537.log`
- `logs/succ-table1-table-15898540.log`

## 结论与下一步

1. **eval 子集 vs 满采样**：eval 子集 6-task mean Acc(0.65)=**0.559**；Table1 扩展 7-task mean **0.407**（overlap-3 **0.457**）。满 100 行/task 更能反映真实难度，不宜仅用 eval 子集宣称全面领先。
2. **DRD2 组合 task 是扩展后亮点**（Acc 0.70，n=100）；**GSK3B↑ 仍为 0**。
3. **Materialized 2p strict 0.78**（n=510）仍明显优于 v1（0.45），但 mean source Tani 降至 0.52。
4. **数据上限 7/10 task**；后续改进方向：GSK3B 监督、train/eval 难度分层报告。
5. **对照 Unified source-anchored v2**：见 [moledit-sourceanchored-v2.md](../unified-3m/moledit-sourceanchored-v2.md)。
