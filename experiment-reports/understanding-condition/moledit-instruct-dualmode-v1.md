# SUCC MolEdit-Instruct Dual-Mode v1

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + de novo / OOD benchmark） |
| **最后更新** | 2026-06-13 |
| **项目** | `SketchMol-Understanding-Condition` |
| **Job** | `15963804`（`succ-dualmode-v1`，54m29s） |
| **入口** | `submit_univideo_moledit_dualmode_pipeline.sh` |
| **动机** | 旧 checkpoint OOD overall strict **0.114** 几乎被 `logp_high=0.750` 撑起；`rare_combo=0%`，MW/TPSA/RB extreme 全 0 → 缺 zero-source 边界/组合分布 |
| **基线** | v3 attack: [moledit-instruct-v3-attack.md](moledit-instruct-v3-attack.md)；OOD eval: [denovo-ood-benchmark.md](denovo-ood-benchmark.md) |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15963804` | `succ-dualmode-v1`（训练） | **54m29s** | 0 |
| `16006990` | `succ-denovo-ood-dualmode` | **16m09s** | 0 |
| `16006991` | `succ-denovo-2p7p-dualmode` | **17m55s** | 0 |

训练 job 提交 2026-06-11 10:34；因 Nibi 全站维护 pending 至 2026-06-12 18:01 才开始。Benchmark 复测 2026-06-13。

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

## Downstream Benchmark（dualmode checkpoint）

对照旧 checkpoint（v2_fix / v3 attack 写入，job `15959362` / `15959891`）。  
Materializer：`latent_nearest`；OOD 使用 negative guidance scale=2.0。

### 总览

| 评测 | Job | Overall strict | Unique valid | Target latent cos |
| --- | --- | ---: | ---: | ---: |
| 2p7p 旧 | `15959362` | 0.055 | 0.004 | 0.095 |
| **2p7p dualmode** | `16006991` | **0.039** | **0.068** | **0.242** |
| OOD 旧 | `15959891` | 0.114 | 0.014 | 0.089 |
| **OOD dualmode** | `16006990` | **0.127** | **0.199** | **0.238** |

Validity 均为 **1.000**。Latent 对齐与多样性显著改善；strict 指标有升有降。

### OOD（job `16006990`）

| Bucket | 旧 | dualmode | Δ |
| --- | ---: | ---: | ---: |
| forward_extreme | 0.188 | 0.083 | ↓ |
| **rare_combo** | **0.000** | **0.010** (4/400) | 从 0 有信号 |
| **reverse_stimulation** | 0.195 | **0.450** | ↑↑ |
| **Overall** | 0.114 | **0.127** | +0.013 |

| Spec | 旧 | dualmode |
| --- | ---: | ---: |
| reverse_logp4 | 0.230 | **0.560** |
| reverse_mw300 | 0.160 | **0.340** |
| logp_high | **0.750** | 0.210 |
| rb_high | 0.000 | **0.100** |
| qed_high_mw_high | 0.000 | 0.010 |
| mw_high / seven_property_edge | 0.000 | 0.000 |

旧结果几乎被 `logp_high=0.75` 撑起；dualmode 后 **reverse_stimulation 成为主贡献**（90/200 strict），`logp_high` 回落至 0.21。

### De novo 2p–7p（job `16006991`）

| k | 旧 | dualmode |
| --- | ---: | ---: |
| 2p | 0.184 | 0.131 |
| 3p | 0.080 | 0.053 |
| 4p | 0.028 | 0.020 |
| 5p | 0.016 | 0.021 |
| 6p | 0.016 | 0.006 |
| 7p | 0.007 | 0.003 |
| **Overall** | **0.055** | **0.039** |

In-distribution strict 略降，但 unique valid **0.4%→6.8%**（~21→~405 SMILES），mode collapse 明显缓解。

## 结论

1. **Dual-mode 部分达成动机**：`rare_combo` 0%→1%；reverse stim 20%→45%；unique SMILES 数量级提升；target latent cos ~0.24（旧 ~0.09）。
2. **未全面变好**：2p7p overall strict 5.5%→3.9%；forward_extreme 18.8%→8.3%；`logp_high` 75%→21%；`mw_high` / `seven_property_edge` 仍 0。
3. **瓶颈可能在 materializer**：latent 更贴 target、输出更多样，但 `latent_nearest` strict 未同步提升；可试 hybrid rerank 或更大 candidate pool。
4. **MolEdit Table1 尚未复测**：需确认 dualmode 未牺牲 edit 能力。
5. **下一步**：加大 OOD/rare_combo train 权重；调高 negative guidance；Table1 extension on dualmode ckpt。

详见 [denovo-ood-benchmark.md](denovo-ood-benchmark.md)、[denovo-2p7p-benchmark.md](denovo-2p7p-benchmark.md)。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1/
  univideo_molecule/univideo_molecule_generation.pt
  dataset/summary.json
SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v1/benchmark_ours/
SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v1/benchmark_ours/
```

日志：
- `logs/succ-dualmode-v1-15963804.log`
- `logs/succ-denovo-ood-dualmode-16006990.log`
- `logs/succ-denovo-2p7p-dualmode-16006991.log`
