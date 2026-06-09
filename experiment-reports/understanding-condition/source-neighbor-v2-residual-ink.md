# UniVideo Source-Neighbor v2 Residual + Ink

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + materialized benchmark） |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Understanding-Condition` |
| **训练 Job** | `15821981`（`succ-univideo-mol`） |
| **Benchmark Job** | `15821983`（`succ-univideo-bench`） |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15821981` | `succ-univideo-mol` | **54 分 35 秒** | 0 |
| `15821983` | `succ-univideo-bench` | **12 分 28 秒**（依赖训练完成后） | 0 |

训练 job 申请 24h / 80G / H100 40GB MIG；benchmark 为 CPU job（`afterok:15821981`）。

## 目的

在 **source-neighbor** multi-property 数据上重训 UniVideo v2-style pipeline（`image_vae` + `residual` + `pred_x0`），不再评估旧 `multiproperty_100k_v1` 的 canonical v2 产物。

## 配置要点

| 变量 | 值 |
| --- | --- |
| 入口 | `submit_univideo_source_neighbor_v2_pipeline.sh` |
| 输出 | `outputs/univideo_molecule_generation_source_neighbor_v2_residual_ink/` |
| 数据 | `multiproperty_source_neighbor_v1/condition_rows.csv` |
| VLM features | `condition_features_multiproperty_hf_vlm_source_neighbor_v1/` |
| Image VAE | 复用旧 v2 `molecule_image_vae.pt` |
| OCR | 跳过（`prepare` only） |

## 数据集（`dataset/summary.json`）

| 项 | 值 |
| --- | --- |
| train / eval rows | 43,798 / 6,202 |
| edit rows cap | 50,000 |
| source Tanimoto mean | 0.595（min 0.4） |

## 训练结果（eval_latent，n=1000）

| 指标 | 值 |
| --- | --- |
| latent MAE | 0.752 |
| source latent cosine | 0.967 |
| target latent cosine | 0.397 |
| decoded 图像非白像素比 | ~4.4% |

## Materialized Benchmark（job `15821983`，n=1000）

来源：`univideo_molecule/benchmark_materialized_primary_fast/benchmark_report.md`

### 2p–7p strict

| method | 2p | 3p | 4p | 5p | 6p | 7p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `edit_latent_source_similarity_rerank` | **0.496** | 0.272 | 0.086 | 0.073 | 0.111 | 0.000 |
| `edit_latent_source_first_rerank` | 0.521 | 0.301 | 0.109 | 0.091 | 0.167 | 0.000 |
| `source_identity` | 0.465 | 0.257 | 0.023 | 0.018 | 0.000 | 0.000 |
| `source_tanimoto_property_oracle` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SketchMol reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |

### Source-similarity（overall strict@T）

| method | mean Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: |
| `edit_latent_source_similarity_rerank` | 0.960 | **0.349** | 0.340 | 0.316 |
| `edit_latent_source_first_rerank` | 0.971 | 0.376 | 0.348 | 0.316 |
| `source_identity` | 1.000 | 0.317 | 0.317 | 0.317 |

Diagnostics：`edit_latent_source_similarity_rerank` 的 source identity=93%，不再像旧 v2 那样 100% collapse；2p strict 0.496 vs 旧 v2 materialized 0.480。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_source_neighbor_v2_residual_ink/
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/eval_latent/
  univideo_molecule/benchmark_materialized_primary_fast/
```

日志：
- `logs/succ-univideo-mol-15821981.log`
- `logs/succ-univideo-bench-15821983.log`

## 结论与下一步

1. Source-neighbor 重训后 **materialized 2p strict 从 0.48 提升到 ~0.50**，且 rerank 不再与 source_identity 完全等价。
2. 仍远低于 `source_tanimoto_property_oracle` 上界；latent→target cosine ~0.40，生成流还需加强。
3. 可与 Unified 3M MolEdit 线对照；MolEdit materialized benchmark 需用专用 condition_rows（见 unified-3m 报告）。
