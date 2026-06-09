# UniVideo v2 Residual + Ink（image-to-structure）

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练与 OCR benchmark 已完成；materialized benchmark 另见 sibling 报告） |
| **最后更新** | 2026-06-08 |
| **项目** | `SketchMol-Understanding-Condition` |
| **完成时间** | 2026-06-06（训练） / 2026-06-08（OCR benchmark 更新） |

## 目的

UniVideo-style 双流：HF VLM understanding → connector → source-conditioned latent diffusion → 分子图像 → MolScribe → 2p–7p 结构 benchmark。v2 重点修复 blank-image collapse（residual latent + ink-aware VAE）。

## 数据与特征

| 资源 | 路径 |
| --- | --- |
| condition rows | `SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv` |
| VLM features | `SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm/` |
| Qwen VLM | `/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct` |

## 关键配置

| 变量 | 典型值 |
| --- | --- |
| `SUCC_UNIFIED_OUTPUT_DIR` | `SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/` |
| `SUCC_LATENT_BACKEND` | `image_vae` |
| `SUCC_LATENT_TARGET_MODE` | `residual` |
| `SUCC_DIFFUSION_OBJECTIVE` | `pred_x0` |
| `SUCC_SAMPLE_ETA` | `0.0` |
| `SUCC_IMAGE_VAE_INK_LOSS_WEIGHT` | `4.0` |
| `SUCC_EVAL_LIMIT` | `1000` |

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/
  molecule_image_vae/molecule_image_vae.pt
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/eval_latent/
  univideo_molecule/image_structure_benchmark/
```

## 结果摘要（OCR 路径，2026-06-08）

来源：`univideo_molecule/image_structure_benchmark/benchmark_report.md`

| method | validity | 2p strict | OCR present | 备注 |
| --- | ---: | ---: | ---: | --- |
| `univideo_image_vae` | 0.000 | 0.000 | 0.000 | OCR 路径当前未解析出有效 SMILES |
| SketchMol structured reference | — | 0.804 | — | 上界参考 |

Diagnostics：`empty:615; raw_token_fallback:385`（decode 图像 OCR 失败为主因）。

## 结论与下一步

- **OCR 路径**指标不可用（validity 0）；详见已完成报告 [materialized-benchmark-v2-primary-fast.md](materialized-benchmark-v2-primary-fast.md)。
- Materialized 评估（job `15810260`）显示：`edit_latent_*` 与 `source_identity` 等价，generator 当前 collapse 为 source copy（2p strict 0.48，source identity 100%）。
- 下一步：排查 diffusion/connector 训练，或重训 pipeline。
