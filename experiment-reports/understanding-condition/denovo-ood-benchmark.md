# De Novo OOD Property-Design Benchmark（SUCC Ours）

| 字段 | 值 |
| --- | --- |
| **状态** | completed |
| **最后更新** | 2026-06-11 |
| **项目** | `SketchMol-Understanding-Condition` |
| **Job** | `15959891`（`succ-denovo-ood-ours`，11m51s） |
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
Checkpoint：`univideo_molecule_generation_moledit_instruct_v2_fix`（v3 attack 训练）。

## 总体结果（`edit_latent` + negative guidance + `latent_nearest`）

| 指标 | 值 |
| --- | ---: |
| Validity | 1.000 |
| **Overall strict** | **0.114** |
| Unique valid | 0.014 |
| Target latent cosine | 0.089 |
| Latent MAE | 1.070 |

对比 in-distribution de novo（`15959362`）：overall strict **0.055** → OOD **0.114**（略好，仍远低于 baseline 0.98）。

## 按 Bucket

| Bucket | n | strict |
| --- | ---: | ---: |
| forward_extreme | 400 | **0.188** |
| rare_combo | 400 | **0.000** |
| reverse_stimulation | 200 | **0.195** |

## 按 Spec

| Spec | Bucket | strict |
| --- | --- | ---: |
| logp_high | forward_extreme | **0.750** |
| reverse_logp4 | reverse_stimulation | 0.230 |
| reverse_mw300 | reverse_stimulation | 0.160 |
| mw_high | forward_extreme | 0.000 |
| tpsa_high | forward_extreme | 0.000 |
| rb_high | forward_extreme | 0.000 |
| qed_high_mw_high | rare_combo | 0.000 |
| mw_high_logp_low | rare_combo | 0.000 |
| tpsa_high_hbd_low | rare_combo | 0.000 |
| seven_property_edge | rare_combo | 0.000 |

**唯一亮点**：单属性极端 LogP（`logp_high`）strict **0.75**；reverse stimulation 对 LogP 略优于 MW。  
**rare_combo 全灭**（0/400），7-property edge 亦 0。

## 结论

1. **Negative guidance 有微弱信号**：reverse_stimulation（0.195）与 forward 中非 LogP spec（0）相比，reverse LogP 达 0.23。
2. **多属性 / 稀有组合仍是盲区**：rare_combo 0%、2p7p de novo 仅 5.5% strict，与 edit 训练分布不匹配。
3. **Mode collapse 持续**：1000 行仅 ~14 个 unique SMILES，与 2p7p ours 相同根因。
4. **下一步**：de novo 专用训练；调高 `SUCC_OOD_NEGATIVE_GUIDANCE_SCALE`；与 SketchMol OOD baseline（`SketchMolBenchmark` reverse/forward preset）对照。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/denovo_ood_ours_v1/
  denovo_ood_rows.csv / denovo_ood_negative_rows.csv
  denovo_ood_eval.jsonl / denovo_ood_negative_eval.jsonl
  ours_eval_latent/
  benchmark_ours/benchmark_report.md
```

日志：`logs/succ-denovo-ood-ours-15959891.log`
