# De Novo 2p–7p Property-Design Benchmark

| 字段 | 值 |
| --- | --- |
| **状态** | completed（baseline + SUCC ours + dualmode） |
| **最后更新** | 2026-06-13 |
| **项目** | `SketchMol-Understanding-Condition` |
| **任务** | 无 source 的 de novo 多属性分子设计（2–7 property） |
| **数据** | `multiproperty_source_neighbor_v1/molecule_database.csv` |

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15958481` | `succ-denovo-2p7p`（baseline） | **1h08m** | 0 |
| `15959362` | `succ-denovo-2p7p-ours`（SUCC UniVideo，v2_fix ckpt） | **13m46s** | 0 |
| `16006991` | `succ-denovo-2p7p-dualmode`（dualmode ckpt） | **17m55s** | 0 |

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

## 结论

1. **Edit-only checkpoint 不适合 zero-source de novo**：v2_fix ckpt target cosine **0.095**，overall strict **5.5%**，unique **0.4%**。
2. **Dual-mode 训练改善 latent 与多样性**，但 strict 仍仅 **3.9%**，远低于 `property_nearest`（0.98）和 SketchMol 参考（0.68–0.80）；瓶颈可能在 `latent_nearest` materializer。
3. **Baseline `property_nearest` 可作为数据和 evaluator 上界**：证明 benchmark 管线与 6000 行目标定义正确。
4. **OOD benchmark** 见 [denovo-ood-benchmark.md](denovo-ood-benchmark.md)；dualmode 训练动机与完整分析见 [moledit-instruct-dualmode-v1.md](moledit-instruct-dualmode-v1.md)。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/denovo_2p7p_v1/
  denovo_2p7p_rows.csv
  benchmark_materialized/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_v1/
  benchmark_ours/benchmark_report.md

SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v1/
  benchmark_ours/benchmark_report.md
```

日志：
- `logs/succ-denovo-2p7p-15958481.log`
- `logs/succ-denovo-2p7p-ours-15959362.log`
- `logs/succ-denovo-2p7p-dualmode-16006991.log`
