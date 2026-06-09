# Unified 3M × MolEdit-Instruct v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + latent eval + materialized benchmark + table metrics） |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Unified-3MDiffusion` |
| **训练 Job** | `15810199`（`smu3m-unified`） |
| **Benchmark Job** | `15837897`（`smu3m-diff-bench`） |
| **Table Job** | `15851992`（`smu3m-moledit-table`，重跑） |

## Slurm 运行记录

| 项 | 值 |
| --- | --- |
| 提交 | 2026-06-08 20:49 |
| 开始 | 2026-06-08 21:49 |
| 结束 | 2026-06-08 21:53 |
| **实际墙钟时间** | **3 分 45 秒** |
| 申请时间上限 | 1:00:00 |
| 申请资源 | 1 CPU，8G 内存，H100 20GB MIG |
| Exit code | 0 |

训练在 economy profile（batch=512，train_limit=50k，5 epochs）下较快完成；1 小时上限未打满。

## 目的

在 MolEdit-Instruct enhanced splits 上训练 Unified 3M（alignment → edit tokens → latent diffusion），并与 MolEditRL 表格指标对齐。

## 数据

```text
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv
```

`DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule`

本次导出数据集（`dataset/summary.json`）：

| 项 | 值 |
| --- | --- |
| train rows | 60,000（含 18,266 description pretrain + 50k edit cap） |
| eval rows | 58,266 |
| edit source Tanimoto mean | 0.708（min 0.65） |
| unique source / target SMILES | 81,824 / 98,841 |

## 提交命令

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
export SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_pipeline.sh
```

## 关键配置

| 变量 | 值 |
| --- | --- |
| `SMU3M_DATASET_MODE` | `moledit` |
| `SMU3M_OUTPUT_DIR` | `SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/` |
| `SMU3M_EPOCHS` | 5 |
| `SMU3M_TRAIN_LIMIT` | 50000 |
| `SMU3M_EVAL_LIMIT` | 1000 |
| `SMU3M_RESOURCE_PROFILE` | economy |
| `SMU3M_DIFFUSION_TARGET` | residual |
| GPU | NVIDIA H100 80GB HBM3 MIG 2g.20gb |

## 产出路径

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/
  dataset/summary.json
  alignment/alignment_model.pt
  edit_condition_tokens/edit_condition_connector.pt
  latent_diffusion/latent_diffusion_generation.pt
  eval_latent/metrics.json
  eval_latent/predictions.csv
  benchmark_materialized_primary_fast/   # job 15837897
  moledit_table_metrics/                 # job 15851992
```

日志：`SketchMol-Unified-3MDiffusion/logs/smu3m-unified-15810199.log`

## 训练曲线（epoch 5）

| 阶段 | 末 epoch train loss | 备注 |
| --- | ---: | --- |
| alignment | 0.372 | 5 epochs |
| edit connector | 1.313 | 5 epochs |
| latent diffusion | 0.268 | diffusion_target_mae≈0.236 |

Diffusion 末 epoch：`source_property_worse_rate`≈0.45%，`source_fingerprint_worse_rate`≈0.55%。

## 结果摘要（eval_latent，n=1000）

来源：`eval_latent/metrics.json`

### Overall

| 指标 | 值 |
| --- | ---: |
| latent MAE | 0.262 |
| target fingerprint cosine | 0.751 |
| source→target fingerprint cosine | 0.840 |
| target property MAE | 9.65 |
| **property MAE beats source** | **87.2%** |
| property MAE beats prior | 30.1% |
| property MAE gain vs source | +8.74 |
| fingerprint beats source | 2.2% |

### 按性质数量（property MAE beats source）

| #props | rows | beats source |
| --- | ---: | ---: |
| 1 | 442 | 91.4% |
| 2 | 171 | 90.6% |
| 3 | 39 | 100% |
| 4 | 46 | 78.3% |
| 5 | 93 | 80.6% |
| 6 | 103 | 74.8% |
| 7 | 106 | 81.1% |

### Benchmark / 表格指标（Slurm，2026-06-09）

| Job | 名称 | 结果 |
| --- | --- | --- |
| `15822025` | `smu3m-diff-bench` | **空结果**（condition rows 错配） |
| `15822029` | `smu3m-moledit-table` | **空结果**（依赖空 benchmark） |
| `15837897` | `smu3m-diff-bench` | **完成**（~4h，1000 eval rows） |
| `15837898` | `smu3m-moledit-table` | **失败**（`task_key` 命名冲突 + 无 TDC） |
| `15851992` | `smu3m-moledit-table` | **完成**（修复后重跑） |

**Materialized benchmark**（`15837897`，`edit_latent_source_similarity_rerank`，4p–7p）：

| 指标 | 值 |
| --- | ---: |
| 4p strict | 0.279 |
| 5p strict | 0.286 |
| 6p strict | 0.244 |
| 7p strict | 0.158 |
| mean source Tanimoto | 0.158 |
| strict@0.4 | 0.001 |

与 latent eval 一致：property 方向有信号，但 **source 保真度极低**（mean Tani≈0.16），远低于 oracle（0.708）。

**MolEdit Table Metrics**（`15851992`，RDKit SA fallback + skip-task 缺 TDC 的 GSK3B/DRD2）：

| task | Acc_all(0.65) | Acc_all(0.15) | n |
| --- | ---: | ---: | ---: |
| Rotbonds↓ | 0 | 0 | 3 |
| MW↑ | 0 | 0 | 1 |
| SA↓ | 0 | 0.333 | 15 |

完整 Table1 需安装 PyTDC（GSK3B/DRD2 oracle）。

**与 MolEditRL 对照**：

MolEditRL paper Table1 baseline 已记录在
[MolEditRL Reference Comparison](../moleditrl-baseline-comparison.md)。Unified 3M
当前只和其中 3 个任务有 overlap，且样本数很小，因此只作为方向性比较。

| task | MolEditRL Acc_all(0.65) | Unified 3M Acc_all(0.65) | MolEditRL Acc_all(0.15) | Unified 3M Acc_all(0.15) |
| --- | ---: | ---: | ---: | ---: |
| Rotbonds↓ | 0.634 | 0.000 | 0.830 | 0.000 |
| MW↑ | 0.404 | 0.000 | 0.856 | 0.000 |
| SA↓ | 0.628 | 0.000 | 0.828 | 0.333 |
| **mean over overlap** | **0.555** | **0.000** | **0.838** | **0.111** |

结论：Unified 3M 的 latent/property 指标虽然有信号，但 materialized table
metrics 明显没有跟上；和 MolEditRL 差距比 SUCC 更大。

重跑入口：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_moledit_benchmark.sh
```

## 结论与下一步

1. **训练链路正常**：job `15810199` 3m45s 完成全流程 + latent eval。
2. **性质方向有信号**：87% 样本 property MAE beats source；materialized 4p–7p strict 0.16–0.29。
3. **Source 保真不足**：mean source Tanimoto≈0.16，strict@0.4≈0.001；生成更像 property retrieval 而非 source-conditioned edit。
4. **对照 SUCC**：SUCC MolEdit materialized 2p strict≈0.45，且 table metric overlap mean 为 0.222/0.445（0.65/0.15），明显好于 Unified 3M 的 0.000/0.111；见 [moledit-instruct-v1.md](../understanding-condition/moledit-instruct-v1.md)。
5. **对照 MolEditRL**：MolEditRL overlap mean 为 0.555/0.838（0.65/0.15），当前 Unified 3M 还有较大差距；完整基线见 [MolEditRL Reference Comparison](../moleditrl-baseline-comparison.md)。
