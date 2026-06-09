# Unified 3M × MolEdit-Instruct v1

| 字段 | 值 |
| --- | --- |
| **状态** | running（Slurm 排队/运行中） |
| **最后更新** | 2026-06-08 |
| **项目** | `SketchMol-Unified-3MDiffusion` |
| **Slurm Job** | `15810199`（`smu3m-unified`） |

## 目的

在 MolEdit-Instruct enhanced splits 上训练 Unified 3M（alignment → edit tokens → latent diffusion），并与 MolEditRL 表格指标对齐。

## 数据

```text
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv      # ~2.98M
$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv  # 50k
```

`DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule`

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
| GPU | H100 20GB MIG |

## 产出路径

```text
SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/
  dataset/summary.json
  alignment/alignment_model.pt
  edit_condition_tokens/edit_condition_connector.pt
  latent_diffusion/latent_diffusion_generation.pt
  eval_latent/metrics.json
  benchmark_materialized_primary_fast/   # 训练后另跑 benchmark
  moledit_table_metrics/                 # 表格指标
```

日志：`SketchMol-Unified-3MDiffusion/logs/smu3m-unified-15810199.log`

## 结果摘要

<!-- 训练完成后从 eval_latent/metrics.json 与 benchmark 抄录 -->

| 指标 | 值 |
| --- | --- |
| eval latent metrics | 待填 |
| materialized benchmark | 待填 |
| MolEdit 2p–7p 表格 | 待填 |

## 训练后评估

```bash
# materialized benchmark
SMU3M_OUTPUT_DIR=SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1 \
bash SketchMol-Unified-3MDiffusion/scripts/submit_unified_materialized_benchmark.sh

# MolEditRL 表格指标
python SketchMol-Unified-3MDiffusion/scripts/evaluate_moledit_table_metrics.py \
  --reference "$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv" \
  --predictions SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/benchmark_materialized_primary_fast/benchmark_decoded.csv \
  --method edit_latent_source_similarity_rerank \
  --output-dir SketchMol-Unified-3MDiffusion/outputs/unified_generation_moledit_instruct_v1/moledit_table_metrics
```

## 结论与下一步

- （待填）
