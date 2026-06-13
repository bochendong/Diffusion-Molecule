# SUCC MolEdit-Instruct Dual-Mode v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练） |
| **最后更新** | 2026-06-12 |
| **项目** | `SketchMol-Understanding-Condition` |
| **Job** | `15963804`（`succ-dualmode-v1`，54m29s） |
| **入口** | `submit_univideo_moledit_dualmode_pipeline.sh` |
| **动机** | 旧 checkpoint OOD overall strict **0.114** 几乎被 `logp_high=0.750` 撑起；`rare_combo=0%`，MW/TPSA/RB extreme 全 0 → 缺 zero-source 边界/组合分布 |
| **基线** | v3 attack: [moledit-instruct-v3-attack.md](moledit-instruct-v3-attack.md)；OOD eval: [denovo-ood-benchmark.md](denovo-ood-benchmark.md) |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15963804` | `succ-dualmode-v1` | **54m29s** | 0 |

提交 2026-06-11 10:34；因 Nibi 全站维护（`NibiMaintenanceJune`）pending 至 2026-06-12 18:01 才开始。

## Dual-mode 数据混合

在 Table1 MolEdit edit 之外，训练 pack 额外混入 de novo 2p–7p 与 OOD rows：

| Split | MolEdit | De novo 2p–7p | OOD | 合计 |
| --- | ---: | ---: | ---: | ---: |
| **train** | 37,374 | 3,000（500/property-count） | 2,000（200/spec） | **42,374** |
| **eval** | 395 | 6,000（1000/property-count） | 1,000（100/spec） | **7,395** |

OOD bucket：`forward_extreme`、`rare_combo`、`reverse_stimulation`（与 [denovo-ood-benchmark.md](denovo-ood-benchmark.md) 同口径）。

Zero-source 行（de novo + OOD）train 共 **5,000** 行，占 train **11.8%**。

## 配置要点（相对 v3 attack）

| 变量 | dualmode v1 | v3 attack |
| --- | --- | --- |
| `SUCC_DATASET_MODE` | `dualmode` | `moledit` |
| train rows | 42,374 | ~37,374 |
| `SUCC_SOURCE_DROPOUT` | **0.30** | 0.05 |
| `SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK` | 12,000 | 12,000 |
| Table1 weight / aux | 5.0 / 0.90 | 5.0 / 0.90 |
| residual_scale / blend | 1.45 / 0.20 | 1.45 / 0.20 |
| stages | 4 + 14 + 6 | 4 + 14 + 6 |
| 输出 | `..._dualmode_v1/` | `..._v2_fix/`（误写） |

复用 ink-aware image VAE：`outputs/univideo_molecule_generation_v2_residual_ink/molecule_image_vae/molecule_image_vae.pt`。

## 训练 loss（history）

| Stage | Epochs | 首 epoch loss | 末 epoch loss |
| --- | ---: | ---: | ---: |
| stage1 connector | 4 | 2.098 | 1.891 |
| stage2 diffusion | 14 | 3.337 | 3.045 |
| stage3 dropout | 6 | 3.035 | 3.006 |

训练收敛正常；stage2→3 loss 略降。

## Eval latent（n=1000，mixed eval pack）

| 指标 | v3 attack（n=395, edit-only） | dualmode v1（n=1000, mixed） |
| --- | ---: | ---: |
| latent MAE | 0.763 | **0.795** |
| source latent cosine | 0.982 | **0.375** |
| target latent cosine | 0.384 | **0.385** |

**source cosine 大幅下降是预期行为**：eval pack 含 6k de novo + 1k OOD 行，大量 zero-source / 低 source 相似度样本；2p 子集 source cosine **≈0**（-0.006）。target cosine 与 v3 基本持平。

### 按 property-count（dualmode eval）

| k | rows | target cos | source cos |
| --- | ---: | ---: | ---: |
| 1 | 284 | 0.369 | 0.986 |
| 2 | 616 | 0.395 | **-0.006** |
| 4–7 | 100 | 0.35–0.39 | 0.99 |

2p 行主要为 zero-source de novo；1p/4p–7p 仍混有 source-conditioned edit。

## 结论与下一步

1. **Dual-mode 训练已完成**，checkpoint 独立于 v2_fix / v3，可直接用于 OOD / de novo 复测。
2. **Latent 层面**：target 对齐未明显劣化；source 贴附在 mixed eval 上被 de novo 行拉低，符合 `source_dropout=0.30` 设计意图。
3. **尚未跑 downstream benchmark**：Table1 / materialized / de novo 2p7p / OOD strict 需另提 job，指向：
   ```bash
   SUCC_OOD_MODEL_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1
   SUCC_DENOVO_MODEL_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1
   ```
4. **关键验收**：`rare_combo` strict 是否从 0% 抬升；MW/TPSA/RB extreme 是否 >0。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1/
  dataset/summary.json
  dataset/univideo_edit_train.jsonl   # 42374 rows
  dataset/univideo_edit_eval.jsonl    # 7395 rows
  dataset/dualmode_pack/{moledit_only,denovo,ood}/
  univideo_molecule/univideo_molecule_generation.pt
  univideo_molecule/metrics.json
  univideo_molecule/eval_latent/metrics.json
SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_dualmode_v1/
```

日志：`logs/succ-dualmode-v1-15963804.log`
