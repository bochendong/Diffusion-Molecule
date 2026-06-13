# De Novo OOD Property-Design Benchmark（SUCC Ours）

| 字段 | 值 |
| --- | --- |
| **状态** | completed（v2_fix ckpt + dualmode ckpt） |
| **最后更新** | 2026-06-13 |
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

## 结论

1. **Dual-mode 训练部分修复 OOD 盲区**：rare_combo 0%→1%；reverse stim 20%→45%；unique 14→199 SMILES。
2. **forward_extreme 退步**：overall 内 `logp_high` 75%→21%；bucket 18.8%→8.3%；`mw_high` 仍 0。
3. **Mode collapse 在 dualmode 上明显缓解**；v2_fix ckpt 1000 行仅 ~14 unique。
4. **仍远低于可用线**（baseline 0.98）；完整训练分析见 [moledit-instruct-dualmode-v1.md](moledit-instruct-dualmode-v1.md)。
5. **下一步**：加大 rare_combo train 权重；试更高 guidance scale；Table1 edit 复测 dualmode ckpt。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/denovo_ood_ours_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v1/
  benchmark_ours/benchmark_report.md
```

日志：
- `logs/succ-denovo-ood-ours-15959891.log`
- `logs/succ-denovo-ood-dualmode-16006990.log`
