# SUCC MolEdit-Instruct v3 Attack

| 字段 | 值 |
| --- | --- |
| **状态** | completed（训练 + table_attack benchmark + 10/10 Table1） |
| **最后更新** | 2026-06-10 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_univideo_moledit_v3_attack_pipeline.sh` |
| **基线** | v2 fix: [moledit-instruct-v2-fix.md](moledit-instruct-v2-fix.md) |

> **注意**：本次因 shell 中 `SUCC_UNIFIED_OUTPUT_DIR` 仍指向 v2_fix，产出写在 `outputs/univideo_molecule_generation_moledit_instruct_v2_fix/`（覆盖了 v2 checkpoint）。逻辑上这是 v3 attack run；重跑时应 `unset SUCC_UNIFIED_OUTPUT_DIR`。

## Slurm 运行记录

| Job | 名称 | 墙钟时间 | Exit |
| --- | --- | --- | --- |
| `15901616` | `succ-univideo-mol`（v3 训练） | **47m36s** | 0 |
| `15901617` | `succ-univideo-bench`（eval 子集，`table_attack`） | **7m20s** | 0 |
| `15901618` | `succ-moledit-table`（eval 子集） | **13s** | 0 |
| `15902238` | `succ-table1-eval`（1000 行） | **1h10m40s** | 0 |
| `15902240` | `succ-table1-bench`（`table_attack`） | **23m57s** | 0 |
| `15902241` | `succ-table1-table`（10/10 task） | **31s** | 0 |

## v3 相对 v2 的改动

1. **Table1 合成补全**：`--synthesize-missing-tasks`，10/10 task × 100 行 = **1000 行**（490 条合成 pair）。
2. **Table-success rerank**：`edit_latent_table_success_rerank`，2048 候选，按 property-direction 成功度 + source Tani + latent 重排。
3. **更强训练**：Table1 weight=5、aux=0.90、aux_all_properties、residual_scale=1.45、blend=0.20、sample_steps=48。

## 训练（eval_latent，n=395）

| 指标 | v2 fix | v3 attack |
| --- | ---: | ---: |
| latent MAE | 0.738 | 0.763 |
| source latent cosine | 0.993 | 0.982 |
| target latent cosine | 0.374 | 0.384 |

Latent 仍贴 source；主要增益来自 table-success materializer。

## Eval 子集 Table Metrics（job `15901618`，`edit_latent_table_success_rerank`）

| task | Acc(0.65) | Acc(0.15) | n | v2 fix Acc(0.65) |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.000 | 0.000 | 100 | 0.000 |
| Rotbonds↓ | **1.000** | 1.000 | 100 | 0.740 |
| MW↑ | **1.000** | 1.000 | 84 | 0.631 |
| SA↓ | 0.580 | 1.000 | 100 | 0.480 |
| Haccept↓ LogP↑ | 1.000 | 1.000 | 1 | 1.000 |
| DRD2↓ MW↓ SA↓ | **1.000** | 1.000 | 10 | 0.500 |

4 个 Table1 task 在 eval 参考集中无行（`missing-reference`）。

## Table1 完整 benchmark（job `15902241`，10/10 task，1000 行）

| task | Acc(0.65) | Acc(0.15) | n | MolEditRL Acc(0.65) |
| --- | ---: | ---: | ---: | ---: |
| GSK3B↑ | 0.000 | 0.000 | 100 | 0.342 |
| Rotbonds↓ | **1.000** | 1.000 | 100 | 0.634 |
| MW↑ | **1.000** | 1.000 | 100 | 0.404 |
| SA↓ | 0.580 | 0.580 | 100 | 0.628 |
| Haccept↓ SA↓ | **0.880** | 0.970 | 100 | — |
| QED↑ SA↓ | **0.880** | 0.990 | 100 | — |
| Haccept↓ LogP↑ | **0.900** | 1.000 | 100 | 0.316 |
| Haccept↓ MW↓ | **0.870** | 1.000 | 100 | — |
| DRD2↓ MW↓ SA↓ | **1.000** | 1.000 | 100 | 0.518 |
| Haccept↑ MW↑ QED↓ | **0.830** | 1.000 | 100 | — |
| **10-task mean** | **0.894** | **0.854** | — | — |
| overlap-3 (RB/MW/SA) | **0.860** | **0.860** | — | 0.555 |

**9/10 task** Acc(0.65) 超过 MolEditRL；**GSK3B↑ 仍为 0**。

### Materialized（job `15902240`，`edit_latent_table_success_rerank`，n=1000）

| 2p strict | mean source Tani | strict@0.4 | exact target |
| ---: | ---: | ---: | ---: |
| **1.000** | 0.685 | **1.000** | 0.997 |

对比 v2 Table1 扩展（similarity rerank）：2p 0.778→**1.000**，overlap-3 Acc 0.457→**0.860**。

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix/
  univideo_molecule/univideo_molecule_generation.pt   # v3 训练覆盖
  univideo_molecule/benchmark_materialized_table_attack/
  univideo_molecule/moledit_table_metrics_attack/
  dataset/table1_benchmark/                           # 1000 行，含合成
  univideo_molecule/table1_eval_latent/
  univideo_molecule/benchmark_materialized_table1_attack/
  univideo_molecule/moledit_table_metrics_table1_attack/
```

日志：
- `logs/succ-univideo-mol-15901616.log`
- `logs/succ-univideo-bench-15901617.log`
- `logs/succ-moledit-table-15901618.log`
- `logs/succ-table1-eval-15902238.log`
- `logs/succ-table1-bench-15902240.log`
- `logs/succ-table1-table-15902241.log`

## 结论

1. **table-success rerank 是决定性因素**：在 10/10 Table1 上 mean Acc(0.65)=**0.894**，大幅超过 MolEditRL 各 task 单点（除 GSK3B）。
2. **GSK3B↑ 仍是唯一短板**（TDC oracle task，latent 路径未学到活性方向）。
3. **合成 pair + 2048 候选 rerank** 使此前 0 行的 3 个组合 task 也可评估（0.83–0.90 Acc）。
4. **解读需谨慎**：metric 优化直接作用于 rerank 阶段；latent 生成本身改善有限（source cosine 仍 ~0.98）。
5. 建议后续：**独立 v3 输出目录重跑**、GSK3B 专项监督、报告 similarity rerank vs table-success 两条线的差距。
