# SUCC Zero-Source Materializer Hybrid Sweep

| 字段 | 值 |
| --- | --- |
| **状态** | **完成**（2026-06-14） |
| **最后更新** | 2026-06-15 |
| **项目** | `SketchMol-Understanding-Condition` |
| **入口** | `submit_denovo_materializer_sweep.sh` |
| **目的** | 判断 zero-source strict 低是否主要来自 `generated_latent -> SMILES` materializer |

## Slurm 运行记录

| Job | 名称 | 墙钟 | Exit | 输出 |
| --- | --- | --- | --- | --- |
| `16056226` | `succ-2p7p-mat-v1` | 1h26m | 0 | `denovo_2p7p_materializer_sweep_v1/` |
| `16056227` | `succ-ood-mat-v1` | 25m | 0 | `denovo_ood_materializer_sweep_v1/` |
| `16056228` | `succ-2p7p-mat-v3` | 1h27m | 0 | `denovo_2p7p_materializer_sweep_v3/` |
| `16056229` | `succ-ood-mat-v3` | 25m | 0 | `denovo_ood_materializer_sweep_v3/` |
| `16056230` | `succ-2p7p-mat-v4` | 1h26m | 0 | `denovo_2p7p_materializer_sweep_v4/` |
| `16056231` | `succ-ood-mat-v4` | 24m | 0 | `denovo_ood_materializer_sweep_v4/` |

## 方法

固定复用已训练 checkpoint，不重新训练：

| Label | Checkpoint |
| --- | --- |
| v1 | `outputs/univideo_molecule_generation_moledit_instruct_dualmode_v1/` |
| v3 | `outputs/univideo_molecule_generation_moledit_instruct_dualmode_v3_guarded/` |
| v4 | `outputs/univideo_molecule_generation_moledit_instruct_dualmode_v4_warmstart_v1/` |

每个 checkpoint 同时跑 2p7p 和 OOD，并输出三种 materializer：

| Method | 说明 |
| --- | --- |
| `latent_nearest` | 旧默认，按 generated latent 最近邻取 candidate |
| `latent_property_rerank` | latent top-4096 shortlist，再按 absolute property strict/distance rerank |
| `property_nearest` | 候选库属性最近邻上界/诊断，不作为最终模型结果 |

默认 hybrid 打分：

```text
score = 100 * strict_fraction - 10 * normalized_property_distance + 1 * latent_score
```

## 2p7p 结果（overall strict，6000 行）

| Checkpoint | Method | Strict | Unique valid | Target latent cos | Hybrid vs latent | Gap to property_nearest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | latent_nearest | 0.039 | 0.068 | 0.242 | — | — |
| v1 | latent_property_rerank | **0.970** | 0.755 | 0.242 | **×24.9** | −0.010 |
| v1 | property_nearest | 0.980 | 0.953 | — | — | — |
| v3 | latent_nearest | 0.029 | 0.005 | 0.580 | — | — |
| v3 | latent_property_rerank | 0.888 | 0.427 | 0.580 | **×31.2** | −0.092 |
| v3 | property_nearest | 0.980 | 0.953 | — | — | — |
| v4 | latent_nearest | 0.012 | 0.004 | 0.584 | — | — |
| v4 | latent_property_rerank | 0.889 | 0.432 | 0.584 | **×75.1** | −0.091 |
| v4 | property_nearest | 0.980 | 0.953 | — | — | — |

按 property count（v1 hybrid vs latent_nearest）：

| 2p | 3p | 4p | 5p | 6p | 7p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 / 0.131 | 0.999 / 0.052 | 0.990 / 0.021 | 0.980 / 0.021 | 0.953 / 0.006 | 0.899 / 0.003 |

（格式：hybrid / latent_nearest）

## OOD 结果（overall strict，1000 行）

| Checkpoint | Method | Overall strict | Unique valid | Target latent cos | Hybrid vs latent | Gap to property_nearest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | latent_nearest | 0.127 | 0.198 | 0.238 | — | — |
| v1 | latent_property_rerank | **0.985** | 0.284 | 0.238 | **×7.8** | **+0.026** |
| v1 | property_nearest | 0.959 | 0.150 | — | — | — |
| v3 | latent_nearest | 0.101 | 0.020 | 0.571 | — | — |
| v3 | latent_property_rerank | 0.960 | 0.084 | 0.571 | **×9.5** | +0.001 |
| v3 | property_nearest | 0.959 | 0.150 | — | — | — |
| v4 | latent_nearest | 0.097 | 0.012 | 0.574 | — | — |
| v4 | latent_property_rerank | 0.956 | 0.084 | 0.574 | **×9.9** | −0.003 |
| v4 | property_nearest | 0.959 | 0.150 | — | — | — |

## 结论

1. **主要瓶颈确认是 materializer**：三种 checkpoint 上 `latent_property_rerank` 相对 `latent_nearest` 提升 **8–75×**（strict），与「latent 对齐尚可、strict 极低」的现象一致。
2. **换 materializer 即可接近上界**：v1 hybrid 在 2p7p 达 **97.0%**（距 `property_nearest` 仅 1pp），OOD 达 **98.5%** 且**超过** `property_nearest`（95.9%）。
3. **更高 latent cos 不等于更好 hybrid**：v3/v4 target cos ~0.57–0.58，但 2p7p hybrid 仅 **~89%**，低于 v1 的 97%；latent shortlist 质量与属性 rerank 的 trade-off 需单独分析。
4. **多样性**：hybrid unique valid 仍低于 `property_nearest`（2p7p ~43–76% vs 95%），但远高于 `latent_nearest`（v3/v4 仅 0.4–0.5%）。
5. **工程落地**：2026-06-15 已将 standalone de novo 2p7p/OOD benchmark 默认切到 `dualmode_v1 + latent_property_rerank`；训练 pipeline 的下游 de novo/OOD 评测也默认使用 hybrid materializer，并写入 `_hybrid` 输出目录，避免覆盖旧 `latent_nearest` 结果。Standalone 验证 job `16072912`/`16072913` 与 sweep `16056226`/`16056227` 数值完全一致（2p7p **0.970**、OOD **0.985**）。
6. **下一步**：训练侧可暂缓 contrastive/RL，优先调 shortlist 大小与 rerank 权重；v3/v4 2p7p hybrid 落后 v1 的原因值得 ablation（shortlist 4096 vs latent 分布）。

## 提交命令（归档）

```bash
export DM_DATA_ROOT=/scratch/bdong/datasets/Diffusion-Molecule
SUCC_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \
bash SketchMol-Understanding-Condition/scripts/submit_denovo_materializer_sweep.sh
```
