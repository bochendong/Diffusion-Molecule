# De Novo OOD Property-Design Benchmark（SUCC Ours）

| 字段 | 值 |
| --- | --- |
| **状态** | completed（v2_fix ckpt + dualmode ckpt） |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **Job** | `15959891`（v2_fix）；`16006990`（dualmode，16m09s） |
| **入口** | `submit_denovo_ood_ours_benchmark.sh` |
| **关联** | in-distribution de novo：[denovo-2p7p-benchmark.md](denovo-2p7p-benchmark.md) |

## 评测设计

三类 OOD（各 100 行/spec，共 **1000** eval 行）：

| Bucket | Specs | n |
| --- | --- | ---: |
| **forward_extreme** | MW:650, LogP:6, TPSA:160, RB:12 | 400 |
| **rare_combo** | QED+MW, MW+LogP, TPSA+HBD, 7-property edge | 400 |
| **reverse_stimulation** | MW:300 / LogP:4 + negative preset | 200 |

采样：`pred = pred_negative + guidance_scale * (pred_positive - pred_negative)`，默认 `SUCC_OOD_NEGATIVE_GUIDANCE_SCALE=2.0`。

## v2_fix Checkpoint（job `15959891`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_v2_fix`（v3 attack 训练写入）。

| 指标 | 值 |
| --- | ---: |
| Validity | 1.000 |
| **Overall strict** | **0.114** |
| Unique valid | 0.014 |
| Target latent cosine | 0.089 |

| Bucket | strict |
| --- | ---: |
| forward_extreme | 0.188 |
| rare_combo | **0.000** |
| reverse_stimulation | 0.195 |

**唯一亮点**：`logp_high` strict **0.75**；rare_combo **0/400**。

## Dual-Mode Checkpoint（job `16006990`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v1`（job `15963804`）。

| 指标 | v2_fix | dualmode | Δ |
| --- | ---: | ---: | ---: |
| Overall strict | 0.114 | **0.127** | +0.013 |
| Unique valid | 0.014 | **0.199** | ↑↑ |
| Target latent cos | 0.089 | **0.238** | ↑↑ |
| Latent MAE | 1.070 | 0.895 | ↓ |

| Bucket | v2_fix | dualmode |
| --- | ---: | ---: |
| forward_extreme | 0.188 | 0.083 |
| **rare_combo** | **0.000** | **0.010** (4/400) |
| **reverse_stimulation** | 0.195 | **0.450** |

| Spec | v2_fix | dualmode |
| --- | ---: | ---: |
| reverse_logp4 | 0.230 | **0.560** |
| reverse_mw300 | 0.160 | **0.340** |
| logp_high | **0.750** | 0.210 |
| rb_high | 0.000 | **0.100** |
| qed_high_mw_high | 0.000 | 0.010 |
| mw_high / seven_property_edge | 0.000 | 0.000 |

Dual-mode 后 overall 由 **reverse_stimulation**（90/200）撑起，而非 `logp_high`。`rare_combo` 从全灭到 4/400。

## Dual-Mode v2 Guarded（job `16019244`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v2_guarded`（job `16018414`，`source_dropout=0.08`，mixed latent）。

| 指标 | v1 dualmode | v2 guarded |
| --- | ---: | ---: |
| Overall strict | **0.127** | 0.052 |
| Unique valid | **0.199** | 0.008 |
| rare_combo | **0.010** | 0.000 |
| reverse_stimulation | **0.450** | 0.100 |

v2 保守 dropout 保留了 Table1 edit，但 **OOD 全面退步**。完整分析见 [moledit-instruct-dualmode-v2-guarded.md](moledit-instruct-dualmode-v2-guarded.md)。

## Dual-Mode v3 Guarded（job `16028796`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded`（job `16027758`，`source_dropout=0.12`，source-free augment，guard 阈值 0.794）。

| 指标 | v1 | v2 | **v3** |
| --- | ---: | ---: | ---: |
| Overall strict | **0.127** | 0.052 | 0.101 |
| Unique valid | **0.199** | 0.008 | 0.020 |
| Target latent cos | 0.238 | 0.238 | **0.571** |
| reverse_stimulation | **0.450** | 0.100 | 0.425 |
| reverse_logp4 | 0.560 | 0.070 | **0.820** |
| rare_combo | **0.010** | 0.000 | 0.000 |

v3 **Table1 guard 通过**，latent 对齐与 `reverse_logp4` 显著提升，但 overall strict 仍略低于 v1。v4 warm-start 见 [moledit-instruct-dualmode-v4-warmstart-v1.md](moledit-instruct-dualmode-v4-warmstart-v1.md)。

## Dual-Mode v4 Warmstart（job `16043206`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1`（v1 warm-start，短 fine-tune）。

| 指标 | v1 | v3 | **v4** |
| --- | ---: | ---: | ---: |
| Overall strict | **0.127** | 0.101 | 0.097 |
| Unique valid | **0.199** | 0.020 | 0.012 |
| Target latent cos | 0.238 | 0.571 | **0.574** |
| reverse_logp4 | 0.560 | **0.820** | 0.060 |
| reverse_mw300 | 0.340 | 0.030 | **0.500** |
| forward_extreme bucket | 0.083 | 0.040 | **0.102** |

Warm-start **未**恢复 v1 overall strict；spec 间 trade-off 加剧。

## SUCC Hybrid Materializer Promotion（job `16056227`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v1`。<br>
Materializer：`latent_property_rerank`（latent top-4096 shortlist，再按 absolute property strict/distance rerank）。

| 方法 | Overall strict | Unique valid | Target latent cosine |
| --- | ---: | ---: | ---: |
| v1 `latent_nearest` | 0.127 | 0.198 | 0.238 |
| v1 `latent_property_rerank` | **0.985** | 0.284 | 0.238 |
| `property_nearest` 上界 | 0.959 | 0.150 | — |

2026-06-15 起，standalone OOD ours benchmark 默认切到
`dualmode_v1 + latent_property_rerank`，输出目录默认使用
`outputs/denovo_ood_ours_dualmode_v1_hybrid/`。

### Standalone 默认验证（job `16072913`）

提交命令：`submit_denovo_ood_ours_benchmark.sh`（无额外 env 覆盖）。结果与 sweep `16056227` **完全一致**：

| Overall strict | Unique valid | Target latent cos | 2p bucket | 7p bucket |
| ---: | ---: | ---: | ---: | ---: |
| **0.985** | 0.284 | 0.238 | 0.970 | 0.940 |

## 结论

1. **Dual-mode 训练部分修复 OOD 盲区**：rare_combo 0%→1%；reverse stim 20%→45%；unique 14→199 SMILES。
2. **forward_extreme 退步**：overall 内 `logp_high` 75%→21%；bucket 18.8%→8.3%；`mw_high` 仍 0。
3. **Mode collapse 在 dualmode 上明显缓解**；v2_fix ckpt 1000 行仅 ~14 unique。
4. **旧 latent materializer 远低于可用线**；hybrid 后 OOD strict 达 **0.985**，已接近/超过候选库属性上界（0.959）。完整训练分析见 [moledit-instruct-dualmode-v1.md](moledit-instruct-dualmode-v1.md)。
5. **Hybrid materializer 已验证并设为默认**：v1 OOD strict 从 **0.127** 提升到 **0.985**，说明 OOD 主瓶颈同样在 materializer；下一步再做 shortlist/weight ablation。v3 完整链见 [moledit-instruct-dualmode-v3-guarded.md](moledit-instruct-dualmode-v3-guarded.md)。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/denovo_ood_ours_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v2_guarded/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v3_guarded/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v4_warmstart_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v1_hybrid/
  benchmark_ours/benchmark_report.md
```

日志：
- `logs/succ-denovo-ood-ours-15959891.log`
- `logs/succ-denovo-ood-dualmode-16006990.log`
- `logs/succ-denovo-ood-ours-16019244.log`
- `logs/succ-denovo-ood-ours-16028796.log`
- `logs/succ-denovo-ood-ours-16043206.log`
- `logs/succ-ood-mat-v1-16056227.log`
- `logs/succ-denovo-ood-ours-16072913.log`
