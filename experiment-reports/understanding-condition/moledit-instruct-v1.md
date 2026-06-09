# SUCC UniVideo × MolEdit-Instruct v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + materialized benchmark + table metrics） |
| **最后更新** | 2026-06-09 |
| **项目** | `SketchMol-Understanding-Condition` |
| **训练 Job** | `15838733`（`succ-univideo-mol`） |
| **Benchmark Job** | `15838734`（`succ-univideo-bench`） |
| **Table Job** | `15844098`（`succ-moledit-table`，重跑） |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15838733` | `succ-univideo-mol` | ~43 min | 0 |
| `15838734` | `succ-univideo-bench` | ~13 min | 0 |
| `15838735` | `succ-moledit-table` | 3s | **失败**（`task_key` 命名冲突） |
| `15844098` | `succ-moledit-table` | ~3s | 0（修复后重跑） |

## 目的

在 MolEdit-Instruct enhanced splits 上训练 UniVideo v2-style pipeline（image_vae + residual + pred_x0），OCR-free materialized benchmark，与 Unified 3M × MolEdit 对齐。

## 配置要点

| 变量 | 值 |
| --- | --- |
| `SUCC_DATASET_MODE` | `moledit` |
| `SUCC_UNIFIED_OUTPUT_DIR` | `outputs/univideo_molecule_generation_moledit_instruct_v1/` |
| `SUCC_CONDITION_FEATURES_DIR` | `outputs/condition_features_moledit_hf_vlm_v1/` |
| Image VAE | 复用 v2 `molecule_image_vae.pt` |
| OCR | 关闭（direct SMILES materialization） |

## 训练结果（eval_latent，n=1000）

| 指标 | 值 |
| --- | ---: |
| latent MAE | 0.735 |
| source latent cosine | 0.991 |
| target latent cosine | 0.380 |

Latent 高度贴近 source（cos≈0.99），与 target cosine 仅 ~0.38。

## Materialized Benchmark（job `15838734`，n=1000）

来源：`univideo_molecule/benchmark_materialized_primary_fast/benchmark_report.md`

### 2p–7p strict

| method | 2p | 3p | 4p | 5p | 6p | 7p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `edit_latent_source_first_rerank` | **0.959** | 0.949 | 0.957 | 0.957 | 0.971 | 0.953 |
| `edit_latent_source_similarity_rerank` | 0.450 | 0.333 | 0.391 | 0.366 | 0.495 | 0.311 |
| `source_identity` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SketchMol reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |

### Source-similarity

| method | mean Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: |
| `edit_latent_source_first_rerank` | 0.700 | 0.972 | 0.971 | 0.030 |
| `edit_latent_source_similarity_rerank` | 0.428 | 0.379 | 0.379 | 0.016 |
| `source_tanimoto_property_oracle` | 0.708 | 1.000 | 1.000 | 0.030 |

Diagnostics：`source_first_rerank` 高 strict（~96%）但 mean Tani≈0.70，接近 oracle 检索上界；`source_similarity_rerank` 2p=0.45、mean Tani≈0.43，更平衡。`source_identity` strict=0（MolEdit 任务需改性质，直接 copy source 不满足）。

## MolEdit Table Metrics（job `15844098`）

来源：`univideo_molecule/moledit_table_metrics/moledit_table_summary.md`

| task | Acc_all(0.65) | Acc_all(0.15) | n |
| --- | ---: | ---: | ---: |
| Rotbonds↓ | 0.333 | 0.667 | 3 |
| MW↑ | 0 | 0 | 1 |
| SA↓ | 0.333 | 0.667 | 15 |

仅覆盖 3 个 Table1 task（无 PyTDC，GSK3B/DRD2 跳过）。

### 与 MolEditRL 对照

MolEditRL paper Table1 baseline 已记录在
[MolEditRL Reference Comparison](../moleditrl-baseline-comparison.md)。
当前 SUCC 只和其中 3 个任务有 overlap，且样本数很小
（Rotbonds↓ n=3，MW↑ n=1，SA↓ n=15），因此下面只作为方向性比较。

| task | MolEditRL Acc_all(0.65) | SUCC Acc_all(0.65) | MolEditRL Acc_all(0.15) | SUCC Acc_all(0.15) |
| --- | ---: | ---: | ---: | ---: |
| Rotbonds↓ | 0.634 | 0.333 | 0.830 | 0.667 |
| MW↑ | 0.404 | 0.000 | 0.856 | 0.000 |
| SA↓ | 0.628 | 0.333 | 0.828 | 0.667 |
| **mean over overlap** | **0.555** | **0.222** | **0.838** | **0.445** |

结论：SUCC 在 Rotbonds↓ / SA↓ 上已有部分成功，但仍明显落后 MolEditRL；
MW↑ 当前完全没打中，是最该优先修的 Table1 单任务。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v1/
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/eval_latent/
  univideo_molecule/benchmark_materialized_primary_fast/
  univideo_molecule/moledit_table_metrics/
```

日志：
- `logs/succ-univideo-mol-15838733.log`
- `logs/succ-univideo-bench-15838734.log`
- `logs/succ-moledit-table-15844098.log`

## 结论与下一步

1. **Materialized 2p strict**：`source_similarity_rerank`≈0.45，优于 Unified 3M（~0，4p+）和 source-neighbor v2（~0.50）。
2. **`source_first_rerank` 虚高**：~96% strict 但 mean Tani≈0.70，接近 property oracle 检索，不宜单独作为生成质量指标。
3. **Latent 仍贴 source**：source cosine≈0.99，需加强 edit signal。
4. **对照 Unified 3M**：SUCC 在当前 table metrics 和 source/edit balance 上都优于 Unified 3M，见 [moledit-instruct-v1.md](../unified-3m/moledit-instruct-v1.md)。
5. **对照 MolEditRL**：SUCC 仍明显落后 MolEditRL；3 个 overlap task 的平均 Acc_all 为 0.222/0.445（0.65/0.15），MolEditRL 为 0.555/0.838。
6. **修复入口**：下一版使用 [moledit-instruct-v2-fix.md](moledit-instruct-v2-fix.md)，重点修 MW↑、Table1-balanced eval 和 active-only direction/delta loss。
