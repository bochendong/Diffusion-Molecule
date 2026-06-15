# SUCC Materializer Random-Shortlist Sanity Sweep

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-15） |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_denovo_materializer_sanity_sweep.sh` |
| **目的** | 检查 hybrid materializer 的 0.970/0.985 是否主要来自 property rerank 本身 |

## Slurm 运行记录

| k | 2p7p Job | OOD Job | 2p7p 墙钟 | OOD 墙钟 |
| ---: | --- | --- | --- | --- |
| 64 | `16075242` | `16075243` | 1h19m | 24m |
| 128 | `16075244` | `16075245` | 1h18m | 24m |
| 256 | `16075246` | `16075247` | 1h19m | 24m |
| 512 | `16075248` | `16075249` | 1h22m | ~24m |
| 1024 | `16075250` | `16075251` | ~1h20m | ~24m |
| 4096 | `16075252` | `16075253` | ~1h26m | ~25m |

Checkpoint：`dualmode_v1`；随机种子 `13`。

## 设计

每个 shortlist size 同时跑 2p7p 和 OOD：

| Method | 说明 |
| --- | --- |
| `latent_property_rerank` | generated latent top-k shortlist，再按 absolute property rerank |
| `random_property_rerank` | 随机 top-k shortlist，再按同一套 absolute property rerank；不使用 generated latent |
| `property_nearest` | 全候选库属性最近邻上界/诊断 |

默认 shortlist size：`64 128 256 512 1024 4096`

## 2p7p overall strict

| k | latent | random | property_nearest | latent − random | latent unique | random unique |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.488 | **0.562** | 0.980 | −0.074 | 0.268 | 0.968 |
| 128 | 0.604 | **0.691** | 0.980 | −0.087 | 0.339 | 0.965 |
| 256 | 0.714 | **0.796** | 0.980 | −0.083 | 0.413 | 0.961 |
| 512 | 0.807 | **0.869** | 0.980 | −0.062 | 0.497 | 0.965 |
| 1024 | 0.883 | **0.926** | 0.980 | −0.043 | 0.593 | 0.966 |
| 4096 | 0.970 | **0.978** | 0.980 | −0.008 | 0.755 | 0.961 |

## OOD overall strict

| k | latent | random | property_nearest | latent − random | latent unique | random unique |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.583 | **0.660** | 0.959 | −0.077 | 0.375 | 0.979 |
| 128 | 0.688 | **0.749** | 0.959 | −0.061 | 0.370 | 0.971 |
| 256 | 0.797 | **0.843** | 0.959 | −0.046 | 0.343 | 0.936 |
| 512 | 0.870 | **0.895** | 0.959 | −0.025 | 0.327 | 0.911 |
| 1024 | 0.926 | **0.940** | 0.959 | −0.014 | 0.311 | 0.817 |
| 4096 | **0.985** | 0.980 | 0.959 | +0.005 | 0.284 | 0.622 |

## 结论

1. **高分主要来自 candidate library + property rerank，而非 generated latent shortlist**：在全部 6 个 k 上，`random_property_rerank` 与 `latent_property_rerank` 差距 ≤8.7pp；k=4096 时 2p7p 几乎重合（0.970 vs 0.978），OOD 仅差 0.5pp。
2. **小 k 下 random 反而高于 latent**：k=64–1024 全程 random strict 更高（2p7p 最多领先 8.7pp，OOD 最多 7.7pp）。说明当前 generated latent **未能**比随机抽样更有效地缩小属性可行域；latent shortlist 可能把真正满足属性的分子挡在 shortlist 外。
3. **property rerank 是主要增益来源**：random@k=4096 已达 2p7p **97.8%**、OOD **98.0%**，无需看 latent 即可复现绝大部分 hybrid 分数。
4. **latent 的唯一潜在价值是 diversity efficiency**：同 strict 下 latent unique 通常低于 random（如 2p7p k=4096：0.755 vs 0.961），但 random 用更大候选池也能追上 strict。
5. **对论文/汇报的含义**：不应把 97%/98.5% strict 直接当作「生成模型 de novo 设计能力」；应拆成 (a) 候选库属性检索上界、(b) rerank policy、(c) latent shortlist 是否优于 random。当前 (c) **不成立**。
6. **下一步**：收紧 benchmark 声称（例如报告 random@k 对照）；或改任务定义（更小候选库、禁止全库 rerank、contrastive shortlist 训练）；ablation rerank 权重与 strict 定义。

## 提交命令（归档）

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_denovo_materializer_sanity_sweep.sh
```

产出：`outputs/denovo_{2p7p,ood}_materializer_sanity_v1_k{64,...,4096}/benchmark_ours/benchmark_report.md`
