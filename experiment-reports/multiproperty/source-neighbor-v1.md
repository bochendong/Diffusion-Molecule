# Multi-Property Source-Neighbor v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（数据集已构建） |
| **最后更新** | 2026-06-08 |
| **项目** | `SketchMol-MultiProperty-EditDataset` |

## 目的

构建 source-neighbor pairing 的 multi-property edit 数据集，供 image / VLM / legacy Unified 3M 并行对照使用（非 MolEdit 主线默认数据）。

## 产出路径

```text
SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/
  condition_rows.csv
  diffusion_edit_manifest.csv
  baseline_variants.csv
  molecule_database.csv
  summary.json
```

## 构建命令

```bash
SMMED_RENDER_IMAGES=0 \
bash SketchMol-MultiProperty-EditDataset/scripts/submit_build_dataset.sh
```

## 下游消费

| 消费者 | 说明 |
| --- | --- |
| HF VLM features | `SUCC_DEFAULT_FEATURES_DIR=.../condition_features_multiproperty_hf_vlm_source_neighbor_v1`（待 export） |
| UniVideo v2 | 当前仍用 `multiproperty_100k_v1`；换 source_neighbor 需新跑 pipeline |
| Legacy Unified 3M | `submit_unified_generation_pipeline.sh` |

## 结果摘要

<!-- 从 summary.json 抄录 row counts、pairing 统计 -->

| 指标 | 值 |
| --- | --- |
| 总行数 | 待填 |
| eval rows | 待填 |

## 结论与下一步

- （待填）
