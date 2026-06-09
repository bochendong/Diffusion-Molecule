# Materialized Benchmark v2（primary_fast，OCR-free）

| 字段 | 值 |
| --- | --- |
| **状态** | completed |
| **最后更新** | 2026-06-08 |
| **项目** | `SketchMol-Understanding-Condition` |
| **Slurm Job** | `15810260`（`succ-univideo-bench`） |
| **依赖实验** | [univideo-v2-residual-ink.md](univideo-v2-residual-ink.md) |

## Slurm 运行记录

| 项 | 值 |
| --- | --- |
| 提交 | 2026-06-08 20:52 |
| 开始 | 2026-06-08 20:55 |
| 结束 | 2026-06-08 21:07 |
| **实际墙钟时间** | **12 分 3 秒** |
| 申请时间上限 | 2:00:00 |
| 申请资源 | 1 CPU，16G 内存，**无 GPU** |

### 为什么只跑了约 12 分钟？

这是 **纯 CPU 的后处理 + 评估 job**，不是训练，也不是 VLM/OCR 推理：

1. **输入已全部就绪**：v2 pipeline 在 6 月已产出 `generated_latents.npy`、`target_latents.npy` 和 `image_path.csv`；本 job 只做 latent → SMILES materialization 与 RDKit 性质评估。
2. **无 GPU、无大模型**：`submit_univideo_materialized_benchmark.sh` 默认不申请 GPU；不走 Qwen-VL，也不跑 MolScribe。
3. **工作量有限**：对 5000 行做 latent nearest-neighbor materialization（5 种 method），再在 1000 条 eval 子集上算 2p–7p strict 与 source Tanimoto 指标；256 候选 rerank 的 cosine 检索在 CPU 上可接受。
4. **2 小时是保守上限**：脚本 `primary_fast` profile 默认 `SUCC_MATERIALIZED_SLURM_TIME=02:00:00`，给 rerank / 大 CSV 留 headroom；本次数据规模下远未打满。

若将来 eval 扩到全量 50k 或提高 `SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES`，墙钟时间会相应变长，但仍应显著短于 GPU 训练 job。

## 目的

在 canonical v2 的 eval latents 上跑 OCR-free materialized benchmark，与 `SketchMol-Unified-3MDiffusion` testing 方法对齐；不依赖 MolScribe 环境。

## 输入

```text
SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink

univideo_molecule/image_structure_benchmark/image_path.csv
univideo_molecule/eval_latent/generated_latents.npy
univideo_molecule/eval_latent/target_latents.npy
```

Materialization：`5000` 行；benchmark 汇总按 **1000** 条 eval 行分层（2p–7p）。

## 提交命令

```bash
export SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python
export SUCC_UNIFIED_OUTPUT_DIR=SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink
export SUCC_MATERIALIZED_BENCHMARK_PROFILE=primary_fast
bash SketchMol-Understanding-Condition/scripts/submit_univideo_materialized_benchmark.sh
```

## Profile：primary_fast 对照方法

```text
source_identity
source_tanimoto_property_oracle
edit_latent_source_first_rerank
edit_latent_source_similarity_rerank
target_oracle
```

## 产出路径

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/
  univideo_molecule/benchmark_materialized_primary_fast/
    benchmark_report.md
    benchmark_summary.csv
    benchmark_decoded.csv
    target_molecules_direct.csv
```

日志：`SketchMol-Understanding-Condition/logs/succ-univideo-bench-15810260.log`

## 结果摘要

来源：`benchmark_materialized_primary_fast/benchmark_report.md`（2026-06-08）

### 2p–7p strict（validity = 1.0 全方法）

| method | 2p | 3p | 4p | 5p | 6p | 7p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `edit_latent_source_similarity_rerank` | 0.480 | 0.222 | 0.028 | 0.062 | 0.053 | 0.000 |
| `edit_latent_source_first_rerank` | 0.480 | 0.222 | 0.028 | 0.062 | 0.053 | 0.000 |
| `source_identity` | 0.480 | 0.222 | 0.028 | 0.062 | 0.053 | 0.000 |
| `source_tanimoto_property_oracle` | 0.883 | 0.801 | 0.809 | 0.846 | 0.658 | 0.727 |
| `target_oracle` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |

### Source-similarity 约束（overall strict@T）

| method | mean Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: |
| `edit_latent_source_similarity_rerank` | 1.000 | 0.299 | 0.299 | 0.299 |
| `source_identity` | 1.000 | 0.299 | 0.299 | 0.299 |
| `source_tanimoto_property_oracle` | 0.676 | 0.838 | 0.543 | 0.043 |
| `target_oracle` | 0.580 | 0.802 | 0.541 | 0.036 |

### Diagnostics（要点）

| method | SMILES present | valid | source identity | exact target | unique valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `edit_latent_source_similarity_rerank` | 1.000 | 1.000 | **1.000** | 0.000 | 0.109 |
| `source_tanimoto_property_oracle` | 1.000 | 1.000 | 0.111 | 0.746 | 0.106 |
| `target_oracle` | 1.000 | 1.000 | 0.000 | 1.000 | 0.109 |

Materialization 阶段：`generated_smiles_present_rate=1.0`，`source_identity_rate=0.622`（5000 行 latent 检索层）。

## 结论与下一步

1. **OCR-free 链路可用**：相对 OCR 路径（validity 0），materialized benchmark 全部方法 validity=1.0，说明评估框架本身正常，问题不在 MolScribe。
2. **模型方法 ≡ source copy**：`edit_latent_*` 与 `source_identity` 指标完全一致；`source_identity=1.0`、mean Tanimoto=1.0，说明 rerank 后仍落回 source SMILES，未产生有效编辑。
3. **与上界差距大**：2p strict 0.48 vs `source_tanimoto_property_oracle` 0.883 vs `target_oracle` 1.0；generator 尚未学到 property-directed edit，candidate library oracle 说明数据/检索空间里有更好答案。
4. **下一步**：优先排查 v2 diffusion/connector 是否 collapse 到 source latent；可对比 Unified 3M MolEdit 训练结果；必要时换 `source_neighbor` 数据重训 UniVideo pipeline。
