# De Novo 2p–7p Property-Design Benchmark

| 字段 | 值 |
| --- | --- |
| **状态** | completed（baseline + SUCC ours + dualmode） |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **任务** | 无 source 的 de novo 多属性分子设计（2–7 property） |
| **数据** | `multiproperty_source_neighbor_v1/molecule_database.csv` |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15958481` | `succ-denovo-2p7p`（baseline） | **1h08m** | 0 |
| `15959362` | `succ-denovo-2p7p-ours`（SUCC UniVideo，v2_fix ckpt） | **13m46s** | 0 |
| `16006991` | `succ-denovo-2p7p-dualmode`（dualmode ckpt） | **17m55s** | 0 |
| `16056226` | `succ-2p7p-mat-v1`（dualmode v1 + hybrid materializer） | **1h26m** | 0 |
| `16072912` | `succ-denovo-2p7p-ours`（standalone 默认 hybrid 验证） | **18m49s** | 0 |

Benchmark 规模：6 档 property count × 1000 行 = **6000 eval 行**；候选库 **87480** 分子（`property_nearest` / `latent_nearest` 各从中检索）。

## Baseline（job `15958481`，`property_nearest`）

OCR-free sanity baseline，不含生成模型，仅从候选库按属性距离选最近分子。

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 | 1.000 | 0.999 | 0.992 | 0.964 | 0.923 | **0.980** |

`target_oracle` 上界均为 1.000（直接返回目标分子）。

产出：`outputs/denovo_2p7p_v1/benchmark_materialized/benchmark_report.md`

## SUCC UniVideo Ours（job `15959362`，`latent_nearest`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_v2_fix`（v3 attack 训练写入）。  
Eval：`source_condition_mode=zero`（无 source 图/指纹），`edit_latent` 采样 + 候选库 latent 最近邻 materialize。

### 2p–7p strict

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.184 | 0.080 | 0.028 | 0.016 | 0.016 | 0.007 | **0.055** |

| 参照 | 2p | 3p | 4p | 5p | 6p | 7p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SketchMol structured ref | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |
| property_nearest baseline | 1.000 | 1.000 | 0.999 | 0.992 | 0.964 | 0.923 |

### Eval latent（n=6000，zero source）

| latent MAE | source latent cosine | target latent cosine |
| ---: | ---: | ---: |
| 1.068 | ~0（无 source） | 0.095 |

### Diagnostics

- Validity：**1.000**（SMILES 均可解析）
- Unique valid：**0.004**（6000 行仅 ~21 个不同分子）→ 严重 mode collapse
- Exact target：**0**

产出：`outputs/denovo_2p7p_ours_v1/benchmark_ours/benchmark_report.md`

## SUCC Dual-Mode v1（job `16006991`，`latent_nearest`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v1`（job `15963804`，训练混入 de novo + OOD rows，`source_dropout=0.30`）。

### 2p–7p strict

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.131 | 0.053 | 0.020 | 0.021 | 0.006 | 0.003 | **0.039** |

### vs 旧 ours（`15959362`）

| 指标 | 旧 v2_fix ckpt | dualmode v1 | Δ |
| --- | ---: | ---: | ---: |
| overall strict | 0.055 | 0.039 | ↓ |
| 2p strict | 0.184 | 0.131 | ↓ |
| unique valid | 0.004 | **0.068** | ↑↑ |
| target latent cos | 0.095 | **0.242** | ↑↑ |
| latent MAE | 1.068 | 0.892 | ↓ |

Strict 略降，但 **mode collapse 大幅缓解**（~21→~405 unique SMILES），latent 更贴 target。

产出：`outputs/denovo_2p7p_ours_dualmode_v1/benchmark_ours/benchmark_report.md`

## SUCC Dual-Mode v2 Guarded（job `16019245`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v2_guarded`（`source_dropout=0.08`，mixed latent）。

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.048 | 0.012 | 0.007 | 0.002 | 0.000 | 0.000 | **0.0115** |

| 指标 | v1 dualmode | v2 guarded |
| --- | ---: | ---: |
| overall strict | **0.039** | 0.0115 |
| unique valid | **0.068** | 0.0015 |

v2 在 2p7p 上 **显著劣于 v1**；Table1 持平 v3 但未换来 de novo 收益。见 [moledit-instruct-dualmode-v2-guarded.md](moledit-instruct-dualmode-v2-guarded.md)。

## SUCC Dual-Mode v3 Guarded（job `16028797`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded`（Table1 guard passed）。

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.117 | 0.038 | 0.012 | 0.001 | 0.002 | 0.001 | **0.0285** |

| 指标 | v1 | v2 | **v3** |
| --- | ---: | ---: | ---: |
| overall strict | **0.039** | 0.0115 | 0.0285 |
| unique valid | **0.068** | 0.0015 | 0.0053 |
| target latent cos | 0.242 | — | **0.580** |

2p 接近 v1（0.117 vs 0.131），overall 仍低于 v1；latent 对齐最佳。见 [moledit-instruct-dualmode-v3-guarded.md](moledit-instruct-dualmode-v3-guarded.md)。

## SUCC Dual-Mode v4 Warmstart（job `16043207`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1`。

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.052 | 0.015 | 0.003 | 0.001 | 0.000 | 0.000 | **0.0118** |

| 指标 | v1 | v3 | **v4** |
| --- | ---: | ---: | ---: |
| overall strict | **0.039** | 0.0285 | 0.0118 |
| unique valid | **0.068** | 0.0053 | 0.0037 |
| target latent cos | 0.242 | 0.580 | **0.584** |

v4 为三代 2p7p strict **最低**；见 [moledit-instruct-dualmode-v4-warmstart-v1.md](moledit-instruct-dualmode-v4-warmstart-v1.md)。

## SUCC Hybrid Materializer Promotion（job `16056226`）

Checkpoint：`univideo_molecule_generation_moledit_instruct_dualmode_v1`。<br>
Materializer：`latent_property_rerank`（latent top-4096 shortlist，再按 absolute property strict/distance rerank）。

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 | 0.999 | 0.990 | 0.980 | 0.953 | 0.899 | **0.970** | 0.755 |

| 对照 | overall strict | unique valid |
| --- | ---: | ---: |
| v1 `latent_nearest` | 0.039 | 0.068 |
| v1 `latent_property_rerank` | **0.970** | 0.755 |
| `property_nearest` 上界 | 0.980 | 0.953 |

2026-06-15 起，standalone 2p7p ours benchmark 默认切到
`dualmode_v1 + latent_property_rerank`，输出目录默认使用
`outputs/denovo_2p7p_ours_dualmode_v1_hybrid/`。

### Standalone 默认验证（job `16072912`）

提交命令：`submit_denovo_2p7p_ours_benchmark.sh`（无额外 env 覆盖）。结果与 sweep `16056226` **完全一致**：

| 2p | 3p | 4p | 5p | 6p | 7p | overall strict | unique valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 | 0.999 | 0.990 | 0.980 | 0.953 | 0.899 | **0.970** | 0.755 |

## 结论

1. **Edit-only checkpoint 不适合 zero-source de novo**：v2_fix ckpt target cosine **0.095**，overall strict **5.5%**，unique **0.4%**。
2. **Dual-mode 训练改善 latent 与多样性**；旧 `latent_nearest` strict 仅 **3.9%**，但 hybrid materializer 后达到 **97.0%**，说明主要瓶颈在 latent-to-SMILES materialization。
3. **Baseline `property_nearest` 可作为数据和 evaluator 上界**：证明 benchmark 管线与 6000 行目标定义正确。
4. **OOD benchmark** 见 [denovo-ood-benchmark.md](denovo-ood-benchmark.md)；v3 见 [moledit-instruct-dualmode-v3-guarded.md](moledit-instruct-dualmode-v3-guarded.md)。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/denovo_2p7p_v1/
  denovo_2p7p_rows.csv
  benchmark_materialized/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v2_guarded/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v3_guarded/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v4_warmstart_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v1_hybrid/
  benchmark_ours/benchmark_report.md
```

日志：
- `logs/succ-denovo-2p7p-15958481.log`
- `logs/succ-denovo-2p7p-ours-15959362.log`
- `logs/succ-denovo-2p7p-dualmode-16006991.log`
- `logs/succ-denovo-2p7p-ours-16019245.log`
- `logs/succ-denovo-2p7p-ours-16028797.log`
- `logs/succ-denovo-2p7p-ours-16043207.log`
- `logs/succ-2p7p-mat-v1-16056226.log`
- `logs/succ-denovo-2p7p-ours-16072912.log`
