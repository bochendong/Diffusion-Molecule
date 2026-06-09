# Unified 3M × MolEdit-Instruct v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + latent eval）；benchmark / 表格指标 **首次提交失败，待重跑** |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Unified-3MDiffusion` |
| **Slurm Job** | `15810199`（`smu3m-unified`） |

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
  benchmark_materialized_primary_fast/   # 尚未跑
  moledit_table_metrics/                 # 尚未跑
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

| Job | 名称 | 用时 | 结果 |
| --- | --- | --- | --- |
| `15822025` | `smu3m-diff-bench` | 36s | **失败（eval_rows=0）** |
| `15822029` | `smu3m-moledit-table` | 7s | **失败（0 行，依赖空 benchmark）** |

**根因**：`submit_unified_materialized_benchmark.sh` 默认使用 `multiproperty_source_neighbor_v1/condition_rows.csv`，与 MolEdit eval latent 的 `condition_id`（如 `135`）不匹配；`restrict_eval_to_edit_latent_index` 后交集为空。

**修复**：已添加 `export_moledit_benchmark_condition_rows.py`，导出 1000 行 MolEdit condition rows 至 `dataset/moledit_benchmark_condition_rows.csv`。重跑时需显式设置：

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
export SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1
export SMU3M_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python

# 1) 导出 condition rows（若尚未生成）
python SketchMol-Unified-3MDiffusion/scripts/export_moledit_benchmark_condition_rows.py \
  --moledit-eval-split "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv" \
  --edit-latent-index "$SMU3M_OUTPUT_DIR/eval_latent/index.csv" \
  --output-csv "$SMU3M_OUTPUT_DIR/dataset/moledit_benchmark_condition_rows.csv"

# 2) materialized benchmark
export SMMED_CONDITION_ROWS="$SMU3M_OUTPUT_DIR/dataset/moledit_benchmark_condition_rows.csv"
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh

# 3) MolEdit 2p–7p 表格
python SketchMol-Unified-3MDiffusion/scripts/evaluate_moledit_table_metrics.py \
  --reference "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv" \
  --predictions "$SMU3M_OUTPUT_DIR/benchmark_materialized_primary_fast/benchmark_decoded.csv" \
  --method edit_latent_source_similarity_rerank \
  --output-dir "$SMU3M_OUTPUT_DIR/moledit_table_metrics"
```

注意：MolEdit benchmark 在 1000 eval × 93k candidate DB 上 CPU 较慢（可 1h+），建议单独 Slurm job。

## 结论与下一步

1. **训练链路正常**：job `15810199` 3m45s 完成全流程 + latent eval。
2. **性质方向有信号**：87% 样本 property MAE beats source。
3. **Benchmark 待有效重跑**：job `15822025`/`15822029` 因 condition_rows 错配产出空表；用 MolEdit 专用 export 后重提。
4. **对照**：SUCC source-neighbor v2 materialized 已完成（见 [source-neighbor-v2-residual-ink.md](../understanding-condition/source-neighbor-v2-residual-ink.md)）。
